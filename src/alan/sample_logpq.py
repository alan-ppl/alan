import math
from typing import Optional, Union

from .Plate import Plate, tree_values, update_scope
from .BoundPlate import BoundPlate
from .Group import Group
from .utils import *
from .reduce_Ks import reduce_Ks, sample_Ks
from .Split import Split
from .Sampler import Sampler
from .dist import Dist
from .logpq import lp_getter
from .Data import Data

PBP = Union[Plate, BoundPlate]

def logPQ_sample(
    name:Optional[str],
    P: Plate, 
    Q: Plate, 
    sample: dict, 
    inputs_params: dict,
    data: dict,
    extra_log_factors: dict, 
    scope: dict[str, Tensor], 
    active_platedims:list[Dim],
    all_platedims:dict[str: Dim],
    groupvarname2Kdim:dict[str, Tensor],
    varname2groupvarname:dict[str, str],
    sampler:Sampler,
    computation_strategy:Optional[Split],
    indices:dict[str, Tensor],
    N_dim:Dim,
    num_samples:int):

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)
    assert isinstance(sample, dict)
    assert isinstance(inputs_params, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)
    assert isinstance(indices, dict)

    #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
    #by name=None.
    if name is not None:
        active_platedims = [*active_platedims, all_platedims[name]]

    scope = update_scope(scope, inputs_params)
    scope = update_scope(scope, sample)
    
    #all_Ks doesn't include Ks from timeseries.
    lps, all_Ks, _, _ = lp_getter(
        name=name,
        P=P, 
        Q=Q, 
        sample=sample, 
        inputs_params=inputs_params,
        data=data,
        extra_log_factors=extra_log_factors, 
        scope=scope, 
        active_platedims=active_platedims,
        all_platedims=all_platedims,
        groupvarname2Kdim=groupvarname2Kdim,
        varname2groupvarname=varname2groupvarname,
        sampler=sampler,
        computation_strategy=computation_strategy)

    # Index into each lp with the indices we've collected so far
    for i in range(len(lps)):
        for dim in list(set(generic_dims(lps[i])).intersection(set(indices.keys()))):
            lps[i] = lps[i].order(dim)[indices[dim]]


    if len(all_Ks) > 0:
        indices = {**indices, **sample_Ks(lps, all_Ks,N_dim, num_samples)}
        
    for childname, childQ in Q.grouped_prog.items():
        if isinstance(childQ, Plate):
            childP = P.flat_prog[childname]
            assert isinstance(childP, Plate)

            indices = logPQ_sample(
                name=childname,
                P=childP, 
                Q=childQ, 
                sample=Q.grouped_get(sample, childname),
                data=Q.grouped_get(data, childname),
                inputs_params=inputs_params.get(childname),
                extra_log_factors=extra_log_factors.get(childname),
                scope=scope,
                active_platedims=active_platedims,
                all_platedims=all_platedims,
                groupvarname2Kdim=groupvarname2Kdim,
                varname2groupvarname=varname2groupvarname,
                sampler=sampler,
                computation_strategy=computation_strategy,
                indices=indices,
                num_samples=num_samples,
                N_dim = N_dim
            )

    return indices

