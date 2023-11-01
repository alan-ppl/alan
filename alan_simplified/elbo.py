import torch as t
from typing import Union
from .Plate import Plate
from .BoundPlate import BoundPlate
from .SamplingType import SingleSample
from .global2local_Kdims import global2local_Kdims

from .utils import *

PlateBoundPlate = Union[Plate, BoundPlate]

def elbo(P: PlateBoundPlate, Q: PlateBoundPlate, all_platesizes: dict[str, int], data: dict[str, t.Tensor], K: int, reparam:bool):
    #Error checking that P and Q make sense.
    #  data appears in P but not Q.
    #  all_platesizes is complete.

    all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}
    data = named2dim_tensordict(all_platedims, data)

    Q_scope = {}
    all_platedims = {platename: Dim(platename, size) for platename, size in all_platesizes.items()}
    sampling_type = SingleSample

    #Sample from Q
    global_Kdim = Dim('K', K)
    globalK_sample = Q.sample(
        scope={},
        active_platedims=[],
        all_platedims=all_platedims,
        sampling_type=SingleSample,
        Kdim=global_Kdim,
        reparam=reparam,
    )

    localK_sample, groupvarname2Kdim = global2local_Kdims(globalK_sample, global_Kdim)

    #Compute logQ
    logQ = Q.log_prob(
        sample=localK_sample,
        scope={},
        active_platedims=[],
        all_platedims=all_platedims,
        sampling_type=SingleSample,
        groupvarname2Kdim=groupvarname2Kdim,
    )

    #Subtract log K
    logQ = logQ.minus_logK()

    #Compute logP
    logP = P.log_prob(
        sample={**localK_sample, **data},
        scope={},
        active_platedims=[],
        all_platedims=all_platedims,
        sampling_type=SingleSample,
        groupvarname2Kdim=groupvarname2Kdim,
    )

    #Take the difference of logP and logQ
    logPQ = logP.minus_logQ(logQ)

    #Sum out Ks and plates
    return logPQ.sum()



