import torch as t
import torch.nn as nn
from typing import Union

from .Plate import Plate, tensordict2tree, flatten_tree
from .BoundPlate import BoundPlate, named2torchdim_flat2tree
from .SamplingType import SamplingType
from .utils import *
from .checking import check_PQ_plate, check_inputs_params, mismatch_names
from .logpq import logPQ_plate
from .SamplingType import Permutation

from .Sample import Sample

PBP = Union[Plate, BoundPlate]


class Problem(nn.Module):
    def __init__(self, P:BoundPlate, Q:BoundPlate, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        super().__init__()

        if (not isinstance(P, BoundPlate)) or (not isinstance(Q, BoundPlate)):
            raise Exception("P and Q must be `BoundPlate`s, not e.g. `Plate`s.  You can convert just using `bound_plate_P = BoundPlate(plate_P)` if it doesn't have any inputs or parameters")

        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.register_buffer("_device_tensor", t.zeros(()))

        self.P = P
        self.Q = Q
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}
        self.data = tensordict2tree(P.plate, named2dim_dict(data, self.all_platedims))

        #Check names in P matches those in Q+data, and there are no duplicates.
        #Check the structure of P matches that of Q.
        check_PQ_plate(None, P.plate, Q.plate, self.data)
        check_inputs_params(P, Q)

        P.check_deps(self.all_platedims)
        Q.check_deps(self.all_platedims)

    @property
    def device(self):
        return self._device_tensor.device

    def check_device(self):
        if not (self.device == self.P.device and self.device == self.Q.device):
            raise Exception("Device issue: Problem, P and/or Q aren't all on the same device.  The easiest way to make sure everything works is to call e.g. problem.to('cuda'), rather than e.g. P.to('cuda').")

    def sample(self, K: int, reparam:bool=True, sampling_type:SamplingType=Permutation):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        self.check_device()

        sample, groupvarname2Kdim = self.Q._sample(K, reparam, sampling_type, self.all_platedims)

        return Sample(
            problem=self,
            sample=sample,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            reparam=reparam,
        )

    def inputs_params(self):
        flat_named = {
            **self.P.inputs_params_flat_named(), 
            **self.Q.inputs_params_flat_named()
        }
        return named2torchdim_flat2tree(flat_named, self.all_platedims, self.P.plate)
