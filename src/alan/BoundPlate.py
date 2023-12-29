from typing import Optional
import torch as t
import torch.nn as nn
from .utils import *
from .Sampler import Sampler, PermutationSampler
from .Plate import tensordict2tree, Plate, flatten_tree
from .Stores import BufferStore, ParameterStore, ModuleStore
from .moments import moments2raw_moments

def named2torchdim_flat2tree(flat_named:dict, all_platedims, plate):
    flat_torchdim = named2dim_dict(flat_named, all_platedims)
    return tensordict2tree(plate, flat_torchdim)

def expand_named(x, names, all_platesizes):
    extra_plate_shape = [all_platesizes[name] for name in names]
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
    def __init__(self, plate: Plate, inputs=None, opt_params=None, qem_params=None, all_platesizes=None):
        super().__init__()
        self.plate = plate

        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.register_buffer("_device_tensor", t.zeros(()))

        if inputs is None:
            inputs = {}
        if opt_params is None:
            opt_params = {}
        if qem_params is None:
            qem_params = {}
        assert isinstance(inputs, dict)
        assert isinstance(opt_params, dict)
        assert isinstance(qem_params, dict)

        for v in inputs.values():
            if not isinstance(v, t.Tensor):
                raise Exception("Inputs must be provided as a plain named tensor")

        for v in [*opt_params.values(), *qem_params.values()]:
            if not isinstance(v, (t.Tensor, tuple)):
                raise Exception("Parameters must be provided as a plain named tensor, or a tuple of a plain named tensor, and some dimension names to expand")

        if all_platesizes is None:
            for v in [*opt_params.values(), *qem_params.values()]:
                if not isinstance(v, t.Tensor):
                    raise Exception("If you don't provide all_platesizes kwarg, then opt_params and qem_params must be plain named tensors")
            all_platesizes = {}

        for k, v in opt_params.items():
            if isinstance(v, tuple):
                named_tensor, plate_names = v
                opt_params[k] = expand_named(named_tensor, plate_names, all_platesizes)

        for k, v in qem_params.items():
            if isinstance(v, tuple):
                named_tensor, plate_names = v
                qem_params[k] = expand_named(named_tensor, plate_names, all_platesizes)

        inputs_params = {**inputs, **opt_params, **qem_params}

        #Error checking: input, param names aren't reserved
        input_param_names = list(inputs_params.keys())
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


        varname2groupvarname_dist = self.plate.varname2groupvarname_dist()
        groupvarname2platenames = self.plate.groupvarname2platenames()

        #arg here is any argument to a distribution; could be an input_param, but also could be another random variable.
        argname2varnames = {}
        for varname, (_, dist) in varname2groupvarname_dist.items():
            for argname in dist.all_args:
                if argname not in argname2varnames:
                    argname2varnames[argname] = []
                argname2varnames[argname] = varname

        #Error checking: inputs_params are used in sensible places in the plate heirarchy.
        #Specifically, bad things happen if an input_param has more platedims than expected when it is used.
        for input_param_name, tensor in inputs_params.items():
            set_input_param_platenames = set(non_none_names(tensor))
            for varname in argname2varnames[input_param_name]:
                groupvarname, _ = varname2groupvarname_dist[varname]
                var_platenames = groupvarname2platenames[groupvarname]
                if not set_input_param_platenames.issubset(set(var_platenames)):
                    raise Exception(f"{input_param_name} is used for {varname}, but that doesn't make sense as {input_param_name} has plate dimensions {input_param_platedimnames}, while {varname} only has {var_platedimnames}")

        qem_varnames = set(argname2varname[argname] for argname in qem_params)


        self._inputs = BufferStore(inputs)
        self._opt_params = ParameterStore(opt_params)
        self._qem_params = BufferStore(qem_params)
        self._dists  = ModuleStore(plate.dists())
            



        #varname2argnames = plate.varname2argnames()

        #QEM stuff:
        #fundamental datatype:
        #  moving_average_rawmoments:  (varnames, RawMoment) -> Tensor
        #need to:
        #  initialize 
        #    by converting initial conventional to mean parameters
        #    dict mapping paramname to varname
        #    dict mapping varname to the distribution.
        #  update
        #    easy: the (varnames, RawMoment) key in moving_average_rawmoments is all you need
        #  convert to conventional params
        #    conversions class tells you the RawMoments that it needs.
        #    look up the RawMoment in moving_average_rawmoments.

        #self.paramname2varname
        #self.varname2dist



    @property
    def device(self):
        return self._device_tensor.device

    def inputs(self):
        return self._inputs.to_dict()

    def opt_params(self):
        return self._opt_params.to_dict()

    def qem_params(self):
        return self._qem_params.to_dict()

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


