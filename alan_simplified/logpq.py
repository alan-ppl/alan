import math
from typing import Optional

from .Plate import Plate
from .tree import Tree
from .utils import *
from .reduce_Ks import reduce_Ks

def logPQ_plate(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample_data: dict, 
        inputs_params: dict,
        extra_log_factors: dict, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor]
        sampling_type:SamplingType
        splits:dict[str, int]):

    assert isinstance(inputs_params, Tree)
    assert isinstance(extra_log_factors, Tree)

    #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
    #by name=None.
    if name is not None:
        active_platedims = [*active_platedims, all_platedims[name]]


    #We want to pass back just the incoming scope, as nothing outside the plate can see
    #variables inside the plate.  So `scope` is the internal scope, and `parent_scope`
    #is the external scope we will pass back.
    parent_scope = scope
    scope = {**scope, **inputs_params.values}

    assert set(P.keys()) == set(sample_data.keys())

    lps = [*extra_log_factors.values]

    for childname, childsample in sample_data.items():
        childP = P[childname]
        #childQ doesn't necessarily have a distribution if sample_data is data.
        #childQ defaults to None in that case.
        childQ = Q.get(childname) 

        if isinstance(childP, Dist):
            assert isinstance(childQ, (Dist, None, Enumerate))
            assert isinstance(childsample, Tensor)
            method = logPQ_dist
        elif isinstance(childP, Plate):
            assert isinstance(childQ, Plate)
            assert isinstance(childsample, dict)
            method = logPQ_plate
        else isinstance(childP, Group):
            assert isinstance(childQ, Group)
            assert isinstance(childsample, dict)
            method = logPQ_group

        lp, scope = method(
            name=childname,
            P=childP, 
            Q=childQ, 
            sample_data=childsample,
            inputs_params=inputs_params.get[name]
            extra_log_factors=extra_log_factors.get[name],
            scope=scope, 
            active_platedims=active_platedims,
            groupvarname2Kdim=groupname2Kdim,
            sampling_type=sampling_type,
            splits=splits)
        lps.append(lp)

    #Sum over Ks
    lp = reduce_Ks(lps, [groupvarname2Kdim[name] for name in P])

    #Sum over plate dimension if present (remember, if this is a top-layer plate which
    #is signalled by name=None, then there won't be a plate dimension.
    if name is not None:
        lp = reduce_Ks.sum(all_platedims[name])

    return lp, parent_scope

def logPQ_dist(
        name:str,
        P:Plate, 
        Q:Plate, 
        sample_data: dict, 
        inputs_params: None, 
        extra_log_factors: None, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor]
        sampling_type:SamplingType
        splits:dict[str, int]):

    assert isinstance(inputs_params, None)
    assert isinstance(extra_log_factors, None)

    Kdim = groupvarname2Kdim[name]
    all_Kdims = set(groupvarname2Kdim.values())

    lpq = P.log_prob(
        sample=sample,
        scope=scope,
        active_platedims=active_platedims,
        Kdim=Kdim,
    )
    if Q is not None:
        lq = Q.log_prob(
            sample=sample,
            scope=scope,
            active_platedims=active_platedims,
            Kdim=Kdim,
        )
        lq = sampling_type.reduce_log_prob(
            lq=lq,
            Kdim=Kdim,
            all_Kdims=all_Kdims,
            active_platedims=active_platedims
        )

        lpq = lpq - lq - math.log(Kdim.size)
    return lpq


def logPQ_group(
        name:str,
        P:Group, 
        Q:Group, 
        sample_data: dict, 
        inputs_params: None, 
        extra_log_factors: None, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor]
        sampling_type:SamplingType
        splits:dict[str, int]):

    assert isinstance(inputs_params, None)
    assert isinstance(extra_log_factors, None)

    Kdim = groupvarname2Kdim[name]
    all_Kdims = set(groupvarname2Kdim.values())

    scope = {**scope}

    total_logP = 0.
    total_logQ = 0.
    for childname, childsample in sample_data.items():
        childP = P[childname]
        childQ = Q[childname]
        assert isinstance(childP, Dist)
        assert isinstance(childQ, Dist)
        assert isinstance(childsample, Tensor)

        child_logP = childP.log_prob(
            sample=sample,
            scope=scope,
            active_platedims=active_platedims,
            Kdim=Kdim,
        )
        child_logQ = childQ.log_prob(
            sample=sample,
            scope=scope,
            active_platedims=active_platedims,
            Kdim=Kdim,
        )
        total_logP = total_logP + child_logP
        total_logQ = total_logQ + child_logQ

        scope[childname] = childsample

    total_logQ = sampling_type.reduce_log_prob(
        lq=total_logQ,
        Kdim=Kdim,
        all_Kdims=all_Kdims,
        active_platedims=active_platedims
    )

    return total_logP - total_logQ - math.log(Kdim.size)
