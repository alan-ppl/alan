import torch as t
from typing import Union

from .Plate import Plate
from .BoundPlate import BoundPlate
from .SamplingType import SamplingType
from .utils import *
from .tree2 import tensordict2tree
from .checking import check_names, check_PQ_plate
from .logpq import logPQ_plate

PBP = Union[Plate, BoundPlate]


def named2dim_dict(tensors: dict[str, t.Tensor], all_platedims: dict[str, Dim], setting=""):
    result = {}
    for varname, tensor in tensors.items():
        if not isinstance(tensor, t.Tensor):
            raise Exception(f"{varname} in {setting} must be a (named) torch Tensor")

        for dimname in tensor.names:
            if (dimname is not None): 
                if (dimname not in all_platedims):
                    raise Exception(f"{dimname} appears as a named dimension in {varname} in {setting}, but we don't have a Dim for that plate.")
                else:
                    dim = all_platedims[dimname]
                    if dim.size != tensor.size(dimname):
                        raise Exception(f"Dimension size mismatch along {dimname} in tensor {varname} in {setting}.  Specifically, the size provided in all_platesizes is {dim.size}, while the size of the tensor along this dimension is {tensor.size(dimname)}.")

        torchdims = [(slice(None) if (dimname is None) else all_platedims[dimname]) for dimname in tensor.names]
        result[varname] = generic_getitem(tensor.rename(None), torchdims)

    return result


class Problem():
    def __init__(self, P: PBP, Q: PBP, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}

        P_inputs_params_named = P.inputs_params_named()
        Q_inputs_params_named = Q.inputs_params_named()

        check_names(P, Q, data.keys(), P_inputs_params_named.keys(), Q_inputs_params_named.keys())

        data_torchdim = named2dim_dict(data, self.all_platedims)
        self.data = tensordict2tree(P, data_torchdim)

        inputs_params_named = {**P_inputs_params_named, **Q_inputs_params_named}
        inputs_params_torchdim = named2dim_dict(inputs_params_named, self.all_platedims)
        self.inputs_params = tensordict2tree(P, inputs_params_torchdim)

        #Check names in P matches those in Q+data, and there are no duplicates.
        #Check the structure of P matches that of Q.
        check_PQ_plate(None, P, Q, self.data)

        self.P = P
        self.Q = Q

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
            inputs_params=self.inputs_params,
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

        lp, _ = logPQ_plate(
            name=None,
            P=self.P, 
            Q=self.Q, 
            sample=sample,
            inputs_params=self.inputs_params,
            data=self.data,
            extra_log_factors=extra_log_factors,
            scope={}, 
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            split=split)

        return lp
