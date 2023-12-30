from typing import Optional
import torch as t
import torch.nn as nn
from .utils import *
from .Sampler import Sampler, PermutationSampler
from .Plate import tensordict2tree, Plate, flatten_tree
from .Stores import BufferStore, ParameterStore, ModuleStore
from .moments import moments2raw_moments
from .Param import OptParam, QEMParam
from .conversions import conversion_dict

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
    return x.expand(*extra_plate_shape, *x.shape).refine_names(*names, *x.names)

def non_none_names(x):
    return [name for name in x.names if name is not None]

class BoundPlate(nn.Module):
    """
    Binds a Plate to inputs (e.g. film features in MovieLens) and learned parameters
    (e.g. approximate posterior parameters).

    Only makes sense at the very top layer. 

    Inputs must be provided as a named tensor, with names corresponding to platenames.

    You can provide params in two formats
    opt_params = 
    """
    def __init__(self, plate: Plate, inputs=None, all_platesizes=None):
        super().__init__()
        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.register_buffer("_device_tensor", t.zeros(()))

        self.plate = plate

        if all_platesizes is None:
            all_platesizes = {}
        assert isinstance(all_platesizes, dict)

        if inputs is None:
            inputs = {}
        assert isinstance(inputs, dict)

        #Check inputs are all named tensors
        for v in inputs.items():
            if not isinstance(v, t.Tensor):
                raise Exception("Inputs must be provided as a plain named tensor")

        #Check no dimension mismatches for inputs.
        for k, v in inputs.items():
            for name in v.names:
                if name is not None:
                    if v.size(name) != all_platesizes[name]:
                        raise Exception("Dimension mismatch for input {k} along dimension {name}; all_platesizes gives {all_platesizes[name]}, while {k} is {v.size(name)}")

        ###################################
        #### Setting up QEM/Opt Params ####
        ###################################

        groupvarname2platenames = self.plate.groupvarname2platenames()

        opt_params = {}

        #List of varname
        self.qem_list_varname = []
        #List of conversion
        self.qem_list_conversion = []
        #List of lists of rmkeys; outer list corresponds to random variables.
        self.qem_list_rmkeys = []
        #Flat list of rmkeys.
        self.qem_flat_list_rmkeys = []

        #Dict of RawMoments, mapping meanname -> Tesor
        qem_meanname2mom = {}
        #Dict mapping varname, distargname 2 paramname
        self.qem_varname_distargname2paramname = {}

        #We need meanname because we need to put the tensors in a BufferStore, which requires a dict as input.
        self.qem_rmkey2meanname = {}
        self.qem_meanname2rmkey = {}


        for varname, (groupvarname, dist) in self.plate.varname2groupvarname_dist().items():
            platenames = groupvarname2platenames[groupvarname]

            if not dist.qem_dist:
                #Not a QEM distribution, so may contain opt_params.
                for paramname, (distargname, param) in dist.opt_qem_params.items():
                    opt_params[paramname] = expand_named(param.init, platenames, all_platesizes)
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
                    init_conv_dict[distargname] = expand_named(param.init, platenames, all_platesizes)
                init_means = conversion.conv2mean(**init_conv_dict)

                for i, rmkey in enumerate(rmkeys):
                    meanname = f"{varname}_{i}"
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
        self._opt_params = ParameterStore(opt_params)
        self._qem_meanname2mom = BufferStore(qem_meanname2mom)
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
            
    @property
    def device(self):
        return self._device_tensor.device

    def inputs(self):
        return self._inputs.to_dict()

    def opt_params(self):
        return self._opt_params.to_dict()

    def qem_params(self):
        """
        Converts moving averages in self.qem_moving_average_moments to a flat dict mapping
        paramname -> conventional parameter
        """
        meanname2mom = self._qem_meanname2mom.to_dict()
        result = {}
        for varname, conversion, rmkeys in zip(self.qem_list_varname, self.qem_list_conversion, self.qem_list_rmkeys):
            means = [meanname2mom[self.qem_rmkey2meanname[rmkey]] for rmkey in rmkeys]
            conv_dict = conversion.mean2conv(*means)
            for distargname, tensor in conv_dict.items():
                paramname = self.qem_varname_distargname2paramname[varname, distargname]
                result[paramname] = tensor
        return result

    def _update_qem_params(self, lr, samp_marg_is):
        rmkey_list = self.qem_flat_list_rmkeys
        if 0 < len(rmkey_list):
            new_moment_list = samp_marg_is._moments_uniform_input(rmkey_list)
            for rmkey, new_moment in zip(rmkey_list, new_moment_list):
                meanname = self.qem_rmkey2meanname[rmkey]

                tensor = getattr(self._qem_meanname2mom, meanname)
                tensor.mul_(1-lr).add_(new_moment, alpha=lr)


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
    
    def check_deps(self, all_platedims:dict[str, Dim]):
        """
        This is run as we enter Problem, and checks that we can sample from P and Q, and hence
        that P and Q make sense.  For instance, checks that dependency structure is valid and
        sizes of tensors are consistent.  Note that this isn't obvious for P, as we never actually
        sample from P, we just evaluate log-probabilities under P.
        """
        self._sample(1, False, PermutationSampler, all_platedims)

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

    def sample(self, all_platesizes:dict[str, int]):
        """
        User-facing sample method, so it should return flat-dict of named Tensors, with no K or N dimensions.
        """
        all_platedims = {platename: Dim(platename, size) for (platename, size) in all_platesizes.items()}
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


