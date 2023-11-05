import torch as t
import torch.nn as nn
from .utils import *
from .SamplingType import SamplingType
from .Plate import tensordict2tree, Plate

class BoundPlate(nn.Module):
    """
    Binds a Plate to inputs (e.g. film features in MovieLens) and learned parameters
    (e.g. approximate posterior parameters).

    Only makes sense at the very top layer.
    """
    def __init__(self, plate: Plate, inputs=None, params=None):
        super().__init__()
        self.plate = plate
        self.prog = plate.prog
        self.sample = plate.sample
        self.groupvarname2Kdim = plate.groupvarname2Kdim
        self.all_prog_names = plate.all_prog_names

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


    def inputs(self):
        return {k: v for (k, v) in self.named_buffers()}

    def params(self):
        return {k: v for (k, v) in self.named_parameters()}

    def inputs_params_flat_named(self):
        """
        Returns a dict mapping from str -> named tensor
        """
        return {**self.inputs(), **self.params()}

    def inputs_params_flat_torchdim(self, all_platedims:dict[str, Dim]):
        """
        Returns a flat dict mapping from str -> torchdim tensor
        """
        return named2dim_dict(self.inputs_params_flat_named(), all_platedims)

    def inputs_params(self, all_platedims:dict[str, Dim]):
        """
        Returns a nested dict mapping from str -> torchdim tensor
        """
        return tensordict2tree(self.plate, self.inputs_params_flat_torchdim(all_platedims))
