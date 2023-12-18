import torch as t
from typing import Union

from .Plate import Plate, tensordict2tree, flatten_tree
from .BoundPlate import BoundPlate, named2torchdim_flat2tree
from .SamplingType import SamplingType
from .utils import *
from .checking import check_PQ_plate, check_inputs_params, mismatch_names
from .logpq import logPQ_plate

from .Sample import Sample

PBP = Union[Plate, BoundPlate]



class Problem():
    def __init__(self, P: BoundPlate, Q: BoundPlate, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        assert isinstance(P, BoundPlate)
        assert isinstance(Q, BoundPlate)

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

    def sample(self, K: int, reparam:bool, sampling_type:SamplingType):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        sample, groupvarname2Kdim = self.Q.sample(K, reparam, sampling_type, self.all_platedims)

        return Sample(
            problem=self,
            sample=sample,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            split=None,
        )

    def inputs_params(self):
        flat_named = {
            **self.P.inputs_params_flat_named(), 
            **self.Q.inputs_params_flat_named()
        }
        return named2torchdim_flat2tree(flat_named, self.all_platedims, self.P.plate)
