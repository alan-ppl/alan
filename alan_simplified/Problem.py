import torch as t
from typing import Union
from .Plate import Plate
from .BoundPlate import BoundPlate

from .utils import *

PBP = Union[Plate, BoundPlate]

class Problem():
    def __init__(self, P: PBP, Q: PBP, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        #TODO: Error checking that P and Q make sense.
        #  data appears in P but not Q.
        #  all_platesizes is complete.
        self.P = P
        self.Q = Q
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}

    def sample(self, K: int, reparam:bool, sampling_type:SamplingType):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        groupvarname2Kdim = Q.groupvarname2Kdim(K)
        scope = {}
        sampling_type = IndependentSample

        sample, _ = self.Q.sample(
            name=None,
            scope={},
            active_platedims=[]
            groupvarname2Kdim=self.groupvarname2Kdim,
            sampling_type=sampling_type,
            reparam=reparam,
        )
        return sample

    def elbo(self, sample):
        pass
        
