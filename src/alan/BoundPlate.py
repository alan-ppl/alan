from typing import Optional
import torch as t
import torch.nn as nn
from .utils import *
from .Sampler import Sampler, PermutationSampler
from .Plate import tensordict2tree, Plate, flatten_tree
from .Stores import BufferStore, ParameterStore, ModuleStore
from .moments import moments_func2name
from .Param import OptParam, QEMParam
from .conversions import conversion_dict
from .Timeseries import Timeseries

def named2torchdim_flat2tree(flat_named:dict, all_platedims, plate):
    flat_torchdim = named2dim_dict(flat_named, all_platedims)
    return tensordict2tree(plate, flat_torchdim)

def expand_named(x, names:list[str], all_platesizes:dict[str, int]):
    names_x = non_none_names(x)

    for name_x in names_x:
        if name_x not in all_platesizes:
            raise Exception(f"{name_x} is specified on a parameter, but is not given in all_platesizes")
        if x.size(name_x) == all_platesizes[name_x]:
            raise Exception(f"{name_x} is given as length {all_platesizes[name_x]} on all_platesizes, but there's a parameter where this dimension is size {x.size(name_x)}")

    for name in names:
        if name not in all_platesizes:
            raise Exception(f"{name} is a plate dimension, but is not given in all_platesizes")

    extra_platenames = list(set(names).difference(names_x))

    extra_plate_shape = [all_platesizes[name] for name in extra_platenames]
    return x.expand(*extra_plate_shape, *x.shape).contiguous().refine_names(*names, *x.names)

def non_none_names(x):
    return [name for name in x.names if name is not None]

class BoundPlate(nn.Module):
    """
    Binds a Plate representing P or Q to platesizes, and initializes parameters specified by OptParam or QEMParam.

    Arguments:
        plate (Plate): 
            The plate specifying P or Q.
        all_platesizes (dict[str, int]):
            Dictionary mapping string platename to integer platesize.

    Keyword Arguments:
        inputs (dict[str, (named) torch.Tensor]):
            Dictionary mapping string input name to input value, as a named ``torch.Tensor``.  This is used to represent e.g. features that the model is conditioned on, but that aren't sampled from the model.  Note that 
        extra_opt_params (dict[str, (named) torch.Tensor]):
            Dictionary mapping string parameter name to initial parameter value, as a named ``torch.Tensor``.  Usually you'd specify parameters to be optimized using OptParam.  But the OptParam approach is slightly restictive, as an OptParam can only be used as a direct argument to a distribution (e.g. `` a = Normal(OptParam(0.), 1.)``, whereas an parameter given here can be used anywhere in the program.

    Inputs or extra_opt_params are specified as named tensors, where the names corresond to the plates (as with data).
    """
    def __init__(self, plate: Plate, all_platesizes:dict[str, int], inputs=None, extra_opt_params=None):
        super().__init__()
        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.register_buffer("_device_tensor", t.zeros(()))

        assert isinstance(plate, Plate)
        self.plate = plate

        if all_platesizes is None:
            all_platesizes = {}
        assert isinstance(all_platesizes, dict)

        for platename in plate.all_platenames():
            if platename not in all_platesizes:
                raise Exception(f"Every plate must have a platesize specified in all_platesizes, but {platename} doesn't have a specified size")
        self.all_platesizes = all_platesizes


        if inputs is None:
            inputs = {}
        assert isinstance(inputs, dict)

        if extra_opt_params is None:
            extra_opt_params = {}
        assert isinstance(extra_opt_params, dict)

        #Check inputs are all named tensors
        inputs_extra_opt_params = {**inputs, **extra_opt_params}
        for k, v in inputs_extra_opt_params.items():
            if not isinstance(v, t.Tensor):
                raise Exception("`inputs` and `extra_opt_params` must be provided as a plain named tensor, but {k} is of type {type(v)}")

        #Check all dimensions used in inputs/extra_opt_params are present in all_platesizes, and match
        for k, v in inputs_extra_opt_params.items():
            for name in v.names:
                if name is not None:
                    if name not in all_platesizes:
                        raise Exception("Dimension name {name} used on input/extra_opt_param {k}, but not provided in all_platesizes")
                    if v.size(name) != all_platesizes[name]:
                        raise Exception("Dimension mismatch for input {k} along dimension {name}; all_platesizes gives {all_platesizes[name]}, while {k} is {v.size(name)}")

        #Check that timeseries inits are in the right place
        check_timeseries(plate)

        #Check that inputs/extra_log_params are used in a place that makes sense with regard to plates.
        groupvarname2platenames = self.plate.groupvarname2platenames()
        varname2groupvarname_dist = self.plate.varname2groupvarname_dist()
        for varname, (groupvarname, dist) in varname2groupvarname_dist.items():
            for argname in dist.all_args:
                if argname in inputs_extra_opt_params:
                    dist_platenames = groupvarname2platenames[groupvarname]
                    input_extra_opt_param_platenames = non_none_names(inputs_extra_opt_params[argname])
                    if not set(input_extra_opt_param_platenames).issubset(dist_platenames):
                        raise Exception(f"{argname} is used on {varname}, which has plates {dist_platenames}, but {argname} has plates {input_extra_opt_param_platenames}")

        ###################################
        #### Setting up QEM/Opt Params ####
        ###################################


        opt_paramname2tensor = extra_opt_params
        self.opt_paramname2trans = {paramname: (lambda x: x) for paramname in opt_paramname2tensor}


        #List of varname
        self.qem_list_varname = []
        #List of conversion
        self.qem_list_conversion = []
        #List of lists of rmkeys; outer list corresponds to random variables.
        self.qem_list_rmkeys = []
        #Flat list of rmkeys.
        self.qem_flat_list_rmkeys = []

        #Dict of mapping meanname -> moving average moment as a tensor.
        qem_meanname2mom = {}
        #Dict mapping paramname to conventional parameter as a tensor
        qem_params = {}
        #Dict mapping varname, distargname 2 paramname
        self.qem_varname_distargname2paramname = {}

        #We need meanname because we need to put the tensors in a BufferStore, which requires a dict as input.
        self.qem_rmkey2meanname = {}
        self.qem_meanname2rmkey = {}


        for varname, (groupvarname, dist) in varname2groupvarname_dist.items():
            platenames = groupvarname2platenames[groupvarname]

            if not dist.qem_dist:
                #Not a QEM distribution, so may contain opt_paramname2tensor.
                for paramname, (distargname, param) in dist.opt_qem_params.items():
                    if paramname in opt_paramname2tensor:
                        raise Exception("OptParam is trying to add parameter named {paramname}, but there's already a parameter with this name")
                    opt_paramname2tensor[paramname] = expand_named(param.init, platenames, all_platesizes)
                    self.opt_paramname2trans[paramname] = param.trans
            else:
            #A QEM distribution, so does not contain opt_params.
                self.qem_list_varname.append(varname)

                conversion = conversion_dict[dist.dist]
                self.qem_list_conversion.append(conversion)

                #Sufficient statistics for a distribution, specified in the form of 
                #rmkeys: tuple[tuple[varname], RawMoment]
                #so we can pass it directly to e.g. marginals.moments.
                rmkeys = [((varname,), mom) for mom in conversion.sufficient_stats]
                self.qem_flat_list_rmkeys = [*self.qem_flat_list_rmkeys, *rmkeys]
                self.qem_list_rmkeys.append(rmkeys)

                #Expand the initial conventional parameters provided to the distribution
                #and use them to compute the initial mean parameters, using conversion.
                init_conv_dict = {}
                for paramname, (distargname, param) in dist.opt_qem_params.items():
                    expanded_conv_param = expand_named(param.init, platenames, all_platesizes)
                    qem_params[paramname] = expanded_conv_param 
                    init_conv_dict[distargname] = expanded_conv_param 
                init_means = conversion.conv2mean(**init_conv_dict)

                for i, rmkey in enumerate(rmkeys):
                    _, rawmoment = rmkey
                    meanname = f"{varname}_{moments_func2name[rawmoment]}"
                    self.qem_rmkey2meanname[rmkey] = meanname
                    self.qem_meanname2rmkey[meanname] = rmkey

                #Put these initial mean parameters into the critical qem_moving_average_moments
                #dict.
                for rmkey, init_mean in zip(rmkeys, init_means):
                    meanname = self.qem_rmkey2meanname[rmkey]
                    qem_meanname2mom[meanname] = init_mean

                #conversion.mean2conv produces a dict:
                #distargname -> Tensor.
                #we need to convert varname, distargname -> paramname
                distargname2paramname = {}
                for paramname, (distargname, param) in dist.opt_qem_params.items():
                    self.qem_varname_distargname2paramname[(varname, distargname)] = paramname

        ################################################
        #### Finished setting up Opt/QEM Params!    ####
        #### Now, assign stuff to Param/BufferStore ####
        #### so it is properly registered on device ####
        ################################################

        self._inputs = BufferStore(inputs)
        self._opt_params = ParameterStore(opt_paramname2tensor) 
        self._qem_params = BufferStore(qem_params)              #dict mapping paramname -> conventional parameter.
        self._qem_means = BufferStore(qem_meanname2mom)  #dict mapping meanname -> moving average mean parameter.
        self._dists  = ModuleStore(plate.varname2dist())

        ###################################
        #### A bit more error checking ####
        ###################################
        
        #Error checking: input, param names aren't reserved
        input_param_names = list(self.inputs_params_flat_named().keys())
        set_input_param_names = set(input_param_names)
        for name in input_param_names:
            check_name(name)
        
        #Error checking: no overlap between names in inputs and params.
        if len(input_param_names) != len(set_input_param_names):
            raise Exception(f"BoundPlate has overlapping names in inputs, opt_params, and/or qem_params")

        #Error checking: no overlap between names in program and in inputs or params.
        prog_names = self.plate.all_prog_names()
        prog_input_param_names_overlap = set_input_param_names.intersection(prog_names)
        if 0 != len(prog_input_param_names_overlap):
            raise Exception(f"The program in BoundPlate has plate/random variable names that overlap with the inputs/params.  Specifically {prog_inputs_param_names_overlap}.")

        #Check that all the dependencies make sense by sampling.
        self.sample()
            
    @property
    def device(self):
        return self._device_tensor.device

    def inputs(self):
        """
        Returns a dictionary of the inputs.
        """
        return self._inputs.to_dict()

    def qem_params(self):
        """
        Returns a dictionary of the parameters learned using QEM.
        """
        return self._qem_params.to_dict()
    
    def qem_means(self):
        """
        Returns a dictionary of the exponential moving average moments used for QEM.
        """
        return self._qem_means.to_dict()

    def opt_params(self):
        """
        Returns a dictionary of the parameters learned by optimization.
        """
        result = {}
        for paramname, tensor in self._opt_params.to_dict().items():
            result[paramname] = self.opt_paramname2trans[paramname](tensor)
        return result

    def _update_qem_convparams(self):
        """
        Converts moving averages in self.qem_moving_average_moments to a flat dict mapping
        paramname -> conventional parameter
        """
        meanname2mom = self.qem_means()

        for varname, conversion, rmkeys in zip(self.qem_list_varname, self.qem_list_conversion, self.qem_list_rmkeys):
            means = [meanname2mom[self.qem_rmkey2meanname[rmkey]] for rmkey in rmkeys]
            conv_dict = conversion.mean2conv(*means)
            for distargname, tensor in conv_dict.items():
                paramname = self.qem_varname_distargname2paramname[varname, distargname]

                getattr(self._qem_params, paramname).copy_(tensor)


    def _update_qem_moving_avg(self, lr, sample, computation_strategy):
        rmkey_list = self.qem_flat_list_rmkeys
        if 0 < len(rmkey_list):
            new_moment_list = sample.moments(rmkey_list, computation_strategy=computation_strategy)
            for rmkey, new_moment in zip(rmkey_list, new_moment_list):
                meanname = self.qem_rmkey2meanname[rmkey]

                tensor = getattr(self._qem_means, meanname)
                assert (tensor.names == new_moment.names)
                tensor.mul_(1-lr).add_(new_moment, alpha=lr)

    def _update_qem_params(self, lr, sample, computation_strategy):
        self._update_qem_moving_avg(lr, sample, computation_strategy)
        self._update_qem_convparams()

    def inputs_params_flat_named(self):
        """
        Returns a dict mapping from str -> named tensor
        """
        return {**self.inputs(), **self.opt_params(), **self.qem_params()}

    def inputs_params(self, all_platedims:dict[str, Dim]):
        return named2torchdim_flat2tree(self.inputs_params_flat_named(), all_platedims, self.plate)

    def sample_extended(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_original_platedims:list[Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:Optional[dict[str, Tensor]],
            extended_data:Optional[dict[str, Tensor]]):
        
        scope = {**scope, **self.inputs_params_flat_named()}

        return self.plate.sample_extended(
            sample,
            name,
            scope,
            inputs_params,
            original_platedims,
            extended_platedims,
            active_original_platedims,
            active_extended_platedims,
            Ndim,
            reparam,
            original_data,
            extended_data)
    
    def _sample(self, K: int, reparam:bool, sampler:Sampler, all_platedims:dict[str, Dim]):
        """
        Internal sampling method.
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        assert isinstance(K, int)
        assert isinstance(reparam, bool)
        assert issubclass(sampler, Sampler)
        #assert isinstance(next(iter(all_platedims.values())), Dim)

        groupvarname2Kdim = self.plate.groupvarname2Kdim(K)

        sample = self.plate.sample(
            name=None,
            scope={},
            inputs_params=self.inputs_params(all_platedims),
            active_platedims=[],
            all_platedims=all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampler=sampler,
            reparam=reparam,
        )

        return sample, groupvarname2Kdim

    def sample(self):
        """
        Returns a single sample from the model, as a flat dictionary of named tensors, where the names correspond to plate dimensions.
        """

        all_platedims = {platename: Dim(platename, size) for (platename, size) in self.all_platesizes.items()}
        set_platedims = list(all_platedims.values())
        torchdim_tree_withK, _ = self._sample(1, False, PermutationSampler, all_platedims)
        torchdim_flatdict_withK = flatten_tree(torchdim_tree_withK)

        torchdim_flatdict_noK = {}
        for k, v in torchdim_flatdict_withK.items():
            K_dims = list(set(generic_dims(v)).difference(set_platedims))
            v = v.order(K_dims)
            v = v.squeeze(tuple(range(len(K_dims))))
            torchdim_flatdict_noK[k] = v.detach()

        return dim2named_dict(torchdim_flatdict_noK)

    def groupvarname2platenames(self):
        return self.plate.groupvarname2platenames()

    def varname2groupvarname(self):
        return self.plate.varname2groupvarname()

def check_timeseries(top_plate:Plate):
    assert isinstance(top_plate, Plate)
    for k, v in top_plate.grouped_prog.items():
        if isinstance(v, Plate):
            check_timeseries_inner(v, top_plate)
        else:
            assert isinstance(v, dict)


def check_timeseries_inner(current_plate:Plate, upper_plate:Plate):
    assert isinstance(current_plate, Plate)
    assert isinstance(upper_plate, Plate)
    upper_varname2groupvarname = upper_plate.varname2groupvarname()

    for k, v in current_plate.grouped_prog.items():
        if isinstance(v, dict):
            #Gather timeseries inits.
            init_groupnames = []
            for gk, gv in v.items():
                if isinstance(gv, Timeseries):
                    init_varname = gv.init
                    if init_varname not in upper_plate.flat_prog:
                        raise Exception("Timeseries must have an initializer that is present in the immediate parent plate.  However, the initializer for timeseries {gk}, i.e. {init_varname} doesn't seem to be present in the immediate parent plate.")

                    init_groupname = upper_varname2groupvarname[init_varname]
                    init_groupnames.append(upper_varname2groupvarname[gv.init])

            #Check all init_groupnames are the same
            if 1 <= len(init_groupnames):
                tsg0 = init_groupnames[0]
                for tsg in init_groupnames[1:]:
                    if tsg != tsg0:
                        raise Exception(f"The initializers for a plate must be grouped in the same way as the timeseries themselves.  However, the initializers for timeseries {list(v.keys())}, on group {k} seemed to be grouped differently")
        else:
            assert isinstance(v, Plate)
            check_timeseries_inner(v, current_plate)
