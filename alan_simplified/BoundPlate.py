import torch as t
import torch.nn as nn
from .utils import *
from .SamplingType import SamplingType

class BoundPlate(nn.Module):
    """
    Binds a Plate to inputs (e.g. film features in MovieLens) and learned parameters
    (e.g. approximate posterior parameters).

    Only makes sense at the very top layer.
    """
    def __init__(self, plate, inputs=None, params=None):
        super().__init__()
        self.plate = plate
        self.prog = plate.prog
        self.sample = plate.sample
        self.groupvarname2Kdim = plate.groupvarname2Kdim

        if inputs is None:
            inputs = {}
        if params is None:
            params = {}
        assert isinstance(inputs, dict)
        assert isinstance(params, dict)

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

    def inputs_params_named(self):
        return {**self.inputs(), **self.params()}

