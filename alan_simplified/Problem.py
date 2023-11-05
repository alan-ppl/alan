import torch as t
from typing import Union

from .Plate import Plate, tensordict2tree
from .BoundPlate import BoundPlate
from .SamplingType import SamplingType
from .utils import *
from .checking import check_PQ_plate, mismatch_names
from .logpq import logPQ_plate

PBP = Union[Plate, BoundPlate]



class Problem():
    def __init__(self, P: PBP, Q: PBP, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        all_names_P     =   P.all_prog_names()
        all_names_Qdata = [*Q.all_prog_names(), *data.keys()]
        mismatch_names("", all_names_P, all_names_Qdata)

        self.P = P
        self.Q = Q
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}
        self.data = tensordict2tree(P, named2dim_dict(data, self.all_platedims))

        #Check names in P matches those in Q+data, and there are no duplicates.
        #Check the structure of P matches that of Q.
        check_PQ_plate(None, P, Q, self.data)


    def sample(self, K: int, reparam:bool, sampling_type:SamplingType):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        groupvarname2Kdim = self.Q.groupvarname2Kdim(K)

        sample, _ = self.Q.sample(
            name=None,
            scope={},
            inputs_params=self.Q.inputs_params(self.all_platedims),
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            reparam=reparam,
        )
        return sample, groupvarname2Kdim

    def elbo(
            self, 
            sample:dict, 
            groupvarname2Kdim:dict[str, Dim], 
            sampling_type:SamplingType, 
            extra_log_factors=None,
            split=None):

        if extra_log_factors is None:
            extra_log_factors = {}
        extra_log_factors = named2dim_dict(extra_log_factors, self.all_platedims)
        extra_log_factors = tensordict2tree(self.P, extra_log_factors)

        lp, _, _ = logPQ_plate(
            name=None,
            P=self.P, 
            Q=self.Q, 
            sample=sample,
            inputs_params_P=self.P.inputs_params(self.all_platedims),
            inputs_params_Q=self.Q.inputs_params(self.all_platedims),
            data=self.data,
            extra_log_factors=extra_log_factors,
            scope_P={}, 
            scope_Q={}, 
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            split=split)

        return lp
