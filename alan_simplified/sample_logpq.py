import math
from typing import Optional

from .Plate import Plate, tree_values, update_scope_inputs_params, update_scope_sample
from .Group import Group
from .utils import *
from .reduce_Ks import reduce_Ks, sample_Ks
from .Split import Split
from .SamplingType import SamplingType
from .dist import Dist
from .logpq import logPQ_dist, logPQ_group, logPQ_plate

def logPQ_sample(
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
    split:Optional[Split],
    indices:dict[str, Tensor],
    N_dim:Dim,
    num_samples:int):

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)
    assert isinstance(sample, dict)
    assert isinstance(inputs_params_P, dict)
    assert isinstance(inputs_params_Q, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)
    assert isinstance(indices, dict)

    
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
            assert isinstance(childQ, (Dist, type(None)))
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
            assert isinstance(dist, Plate)
            

    # Index into each lp with the indices we've collected so far
    for i in range(len(lps)):
        for dim in list(set(generic_dims(lps[i])).intersection(set(indices.keys()))):
            lps[i] = lps[i].order(dim)[indices[dim]]


    if len(all_Ks) > 0:
        indices = {**indices, **sample_Ks(lps, all_Ks,N_dim, num_samples)}



        
    for childname, childP in P.prog.items():
        childQ = Q.prog.get(childname)
        
        if isinstance(childP, Plate):
            assert isinstance(childQ, Plate)
            indices = logPQ_sample(name=childname,
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
            split=split,
            indices=indices,
            num_samples=num_samples,
            N_dim = N_dim)
            
        childsample = sample.get(childname)
        if childsample is not None:
            scope_P = update_scope_sample(scope_P, childname, childP, childsample)


    return indices

