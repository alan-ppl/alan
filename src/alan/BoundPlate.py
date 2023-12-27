from typing import Optional
import torch as t
import torch.nn as nn
from .utils import *
from .Sampler import Sampler, PermutationSampler
from .Plate import tensordict2tree, Plate, flatten_tree

def named2torchdim_flat2tree(flat_named:dict, all_platedims, plate):
    flat_torchdim = named2dim_dict(flat_named, all_platedims)
    return tensordict2tree(plate, flat_torchdim)


class BoundPlate(nn.Module):
    """
    Binds a Plate to inputs (e.g. film features in MovieLens) and learned parameters
    (e.g. approximate posterior parameters).

    Only makes sense at the very top layer.
    """
    def __init__(self, plate: Plate, inputs=None, params=None):
        super().__init__()
        self.plate = plate

        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.register_buffer("_device_tensor", t.zeros(()))

        if inputs is None:
            inputs = {}
        if params is None:
            params = {}
        assert isinstance(inputs, dict)
        assert isinstance(params, dict)

        #Error checking: input, param names aren't reserved
        input_param_names = [*inputs.keys(), *params.keys()]
        for name in input_param_names:
            check_name(name)
        
        #Error checking: no overlap between names in inputs and params.
        inputs_params_overlap = set(inputs.keys()).intersection(params.keys())
        if 0 != len(inputs_params_overlap):
            raise Exception(f"BoundPlate has overlapping names {inputs_params_overlap} in inputs and params")

        #Error checking: no overlap between names in program and in inputs or params.
        prog_names = self.plate.all_prog_names()
        prog_input_params_overlap = set(input_param_names).intersection(prog_names)
        if 0 != len(inputs_params_overlap):
            raise Exception(f"The program in BoundPlate has names that overlap with the inputs/params.  Specifically {prog_inputs_params_overlap}.")

        for name, inp in inputs.items():
            assert isinstance(inp, t.Tensor)
            self.register_buffer(name, inp)
        for name, param in params.items():
            assert isinstance(param, t.Tensor)
            self.register_parameter(name, nn.Parameter(param))

    @property
    def device(self):
        return self._device_tensor.device

    def inputs(self):
        return {k: v for (k, v) in self.named_buffers() if k != "_device_tensor"}

    def params(self):
        return {k: v for (k, v) in self.named_parameters()}

    def inputs_params_flat_named(self):
        """
        Returns a dict mapping from str -> named tensor
        """
        return {**self.inputs(), **self.params()}

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
            device=self.device,
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

    def groupvarname2active_platedimnames(self):
        return self.plate.groupvarname2active_platedimnames([])

    def varname2groupvarname(self):
        return self.plate.varname2groupvarname()


