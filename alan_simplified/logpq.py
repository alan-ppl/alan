import math
from typing import Optional

from .Plate import Plate, tree_values, update_scope_inputs_params, update_scope_sample
from .Group import Group
from .utils import *
from .reduce_Ks import reduce_Ks
from .Split import Split
from .SamplingType import SamplingType
from .dist import Dist
from .Data import Data

def logPQ_plate(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample: dict, 
        inputs_params_P: dict,
        inputs_params_Q: dict,
        data: dict,
        extra_log_factors: dict, 
        scope_P: dict[str, Tensor], 
        scope_Q: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampling_type:SamplingType,
        split:Optional[Split]):

    assert isinstance(sample, dict)
    assert isinstance(inputs_params_P, dict)
    assert isinstance(inputs_params_Q, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)

    lps, all_Ks, scope_P, scope_Q, active_platedims = lp_getter(
        name=name,
        P=P, 
        Q=Q, 
        sample=sample, 
        inputs_params_P=inputs_params_P,
        inputs_params_Q=inputs_params_Q,
        data=data,
        extra_log_factors=extra_log_factors, 
        scope_P=scope_P, 
        scope_Q=scope_Q, 
        active_platedims=active_platedims,
        all_platedims=all_platedims,
        groupvarname2Kdim=groupvarname2Kdim,
        sampling_type=sampling_type,
        split=split)

    #Sum out Ks
    lp = reduce_Ks(lps, all_Ks)

    #Sum over plate dimension if present (remember, if this is a top-layer plate which
    #is signalled by name=None, then there won't be a plate dimension.
    if name is not None:
        lp = lp.sum(active_platedims[-1])

    return lp

def logPQ_dist(
        name:str,
        P:Plate, 
        Q:Optional[Plate], 
        sample: OptionalTensor,
        inputs_params_P: dict,
        inputs_params_Q: dict,
        data: OptionalTensor,
        extra_log_factors: None,
        scope_P: dict[str, Tensor], 
        scope_Q: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampling_type:SamplingType,
        split:Optional[Split]):

    assert isinstance(sample, OptionalTensor)
    assert inputs_params_P is None
    assert inputs_params_Q is None
    assert isinstance(data, OptionalTensor)
    assert extra_log_factors is None

    #we must have either sample or data, but not both.
    assert (sample is None) != (data is None)
    sample_data = sample if sample is not None else data
    #if we have a sample, we must have a Q
    if sample is not None:
        assert Q is not None

    lpq = P.log_prob(sample=sample_data, scope=scope_P)

    if sample is not None:
        Kdim = groupvarname2Kdim[name]
        lq = Q.log_prob(sample=sample, scope=scope_Q)
        lq = sampling_type.reduce_logQ(lq, active_platedims, Kdim)

        lpq = lpq - lq - math.log(Kdim.size)
    return lpq


def logPQ_group(
        name:str,
        P:Group, 
        Q:Group, 
        sample: dict, 
        inputs_params_P: dict,
        inputs_params_Q: dict,
        data: None,
        extra_log_factors: None, 
        scope_P: dict[str, Tensor], 
        scope_Q: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampling_type:SamplingType,
        split:Optional[Split]):


    assert isinstance(sample, dict)
    assert inputs_params_P is None
    assert inputs_params_Q is None
    assert data is None
    assert extra_log_factors is None

    Kdim = groupvarname2Kdim[name]
    all_Kdims = set(groupvarname2Kdim.values())

    #Don't need to copy or update scope_Q, as we have already dumped every variable
    #in the plate into that scope.
    scope_P = {**scope_P}

    total_logP = 0.
    total_logQ = 0.
    for childname, childP in P.prog.items():
        childQ = Q.prog[childname]
        childsample = sample[childname]
        assert isinstance(childP, Dist)
        assert isinstance(childQ, Dist)
        assert isinstance(childsample, Tensor)

        total_logP = total_logP + childP.log_prob(sample=childsample, scope=scope_P)
        total_logQ = total_logQ + childQ.log_prob(sample=childsample, scope=scope_Q)

        scope_P[childname] = childsample

    total_logQ = sampling_type.reduce_logQ(total_logQ, active_platedims, Kdim)

    logPQ = total_logP - total_logQ - math.log(Kdim.size)
    return logPQ


def lp_getter(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample: dict, 
        inputs_params_P: dict,
        inputs_params_Q: dict,
        data: dict,
        extra_log_factors: dict, 
        scope_P: dict[str, Tensor], 
        scope_Q: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampling_type:SamplingType,
        split:Optional[Split]):
    """Traverses Q according to the structure of P collecting log probabilities
    
    """
        #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
    #by name=None.
    if name is not None:
        active_platedims = [*active_platedims, all_platedims[name]]


    #We want to pass back just the incoming scope, as nothing outside the plate can see
    #variables inside the plate.  So `scope` is the internal scope, and `parent_scope`
    #is the external scope we will pass back.
    scope_P = update_scope_inputs_params(scope_P, inputs_params_P)
    scope_Q = update_scope_inputs_params(scope_Q, inputs_params_Q)

    assert set(P.prog.keys()) == set([*sample.keys(), *tree_values(data).keys()])

    lps = list(tree_values(extra_log_factors).values())

    #Dump all the scope for Q directly
    #That allows P and Q to have different orders.
    #Note that we already know Q has a valid order, because we sampled from Q
    for childname, childQ in Q.prog.items():
        childsample = sample[childname]
        scope_Q = update_scope_sample(scope_Q, childname, childQ, childsample)

    for childname, childP in P.prog.items():
        childQ = Q.prog.get(childname) 

        #childQ doesn't necessarily have a distribution if sample_data is data.
        #childQ defaults to None in that case.

        if isinstance(childP, Dist):
            assert isinstance(childQ, (Dist, type(None), Data))
            method = logPQ_dist
        elif isinstance(childP, Plate):
            assert isinstance(childQ, Plate)
            method = logPQ_plate
        else:
            isinstance(childP, Group)
            assert isinstance(childQ, Group)
            method = logPQ_group

        lp = method(
            name=childname,
            P=childP, 
            Q=childQ, 
            sample=sample.get(childname),
            data=data.get(childname),
            inputs_params_P=inputs_params_P.get(childname),
            inputs_params_Q=inputs_params_Q.get(childname),
            extra_log_factors=extra_log_factors.get(childname),
            scope_P=scope_P, 
            scope_Q=scope_Q, 
            active_platedims=active_platedims,
            all_platedims=all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            split=split)
        lps.append(lp)

        childsample = sample.get(childname)
        if childsample is not None:
            scope_P = update_scope_sample(scope_P, childname, childP, childsample)

    #Collect all Ks in the plate
    all_Ks = []
    for varname, dist in Q.prog.items():
        if isinstance(dist, (Dist, Group)):
            all_Ks.append(groupvarname2Kdim[varname])
        else:
            assert isinstance(dist, (Plate, Data))
            
    return lps, all_Ks, scope_P, scope_Q, active_platedims