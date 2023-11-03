import torch as t
import torch.nn as nn
from .utils import *
from .SamplingType import SamplingType

#TODO: error checking:
#  inputs can't overlap with parameters.
#  inputs and parameters can't overlap with random variables in plate.
#  inputs and parameters can't be called "plate"
#  all named dimensions in inputs + parameters must be associated with a plate.


class BoundPlate(nn.Module):
    """
    Binds a Plate to inputs (e.g. film features in MovieLens) and learned parameters
    (e.g. approximate posterior parameters).
    """
    def __init__(self, plate, inputs=None, parameters=None):
        super().__init__()
        self.plate = plate

        if inputs is None:
            inputs = {}
        if parameters is None:
            parameters = {}
        assert isinstance(inputs, dict)
        assert isinstance(parameters, dict)

        for name, inp in inputs.items():
            assert isinstance(inp, t.Tensor)
            self.register_buffer(name, inp)
        for name, param in parameters.items():
            assert isinstance(param, t.Tensor)
            self.register_parameter(name, nn.Parameter(param))

    def inputs(self, all_platedims:dict[str, Dim]):
        return named2dim_tensordict(all_platedims, {k: v for (k, v) in self.named_buffers()})

    def parameters(self, all_platedims:dict[str, Dim]):
        return named2dim_tensordict(all_platedims, {k: v for (k, v) in self.named_parameters()})

    def update_scope(self, scope:dict[str, Tensor], all_platedims:dict[str, Dim]):
        return {**scope, **self.inputs(all_platedims), **self.parameters(all_platedims)}
