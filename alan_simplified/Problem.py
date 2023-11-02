import torch as t
from typing import Union
from .Plate import Plate
from .BoundPlate import BoundPlate
from .SamplingType import SingleSample
from .global2local_Kdims import global2local_Kdims

from .utils import *

PlateBoundPlate = Union[Plate, BoundPlate]

class Problem():
    def __init__(self, P: PlateBoundPlate, Q: PlateBoundPlate, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        self.P = P
        self.Q = Q
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}
        self.data = named2dim_tensordict(self.all_platedims, data)

    def sample(self, K: int, reparam:bool):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        #TODO: Error checking that P and Q make sense.
        #  data appears in P but not Q.
        #  all_platesizes is complete.

        Q_scope = {}
        sampling_type = SingleSample

        #Sample from Q
        global_Kdim = Dim('K', K)
        globalK_sample = self.Q.sample(
            scope={},
            active_platedims=[],
            all_platedims=self.all_platedims,
            sampling_type=SingleSample,
            Kdim=global_Kdim,
            reparam=reparam,
        )

        localK_sample, groupvarname2Kdim = global2local_Kdims(globalK_sample, global_Kdim)

        #Compute logQ
        logQ = self.Q.log_prob(
            sample=localK_sample,
            scope={},
            active_platedims=[],
            all_platedims=self.all_platedims,
            sampling_type=SingleSample,
            groupvarname2Kdim=groupvarname2Kdim,
        )

        #Subtract log K
        logQ = logQ.minus_logK()

        #Compute logP
        logP = self.P.log_prob(
            sample={**localK_sample, **self.data},
            scope={},
            active_platedims=[],
            all_platedims=self.all_platedims,
            sampling_type=SingleSample,
            groupvarname2Kdim=groupvarname2Kdim,
        )

        #Take the difference of logP and logQ
        logPQ = logP.minus_logQ(logQ)

        return Sample(globalK_sample, logPQ)

class Sample():
    def __init__(self, sample, logPQ):
        self.sample = sample
        self.logPQ = logPQ

    def elbo(self):
        return self.logPQ.sum()

    def moments(self, fs):
        pass
