import math
from typing import Optional, Union

from .Plate import Plate, tree_values, update_scope
from .Group import Group
from .Data import Data
from .Timeseries import Timeseries
from .dist import Dist

from .utils import *
from .reduce_Ks import reduce_Ks
from .Split import Split, checkpoint, no_checkpoint
from .Sampler import Sampler

def logPQ_plate(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample: dict, 
        inputs_params: dict,
        data: dict,
        extra_log_factors: dict, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampler:Sampler,
        computation_strategy:Optional[Split]):

    #Returns a tuple of dicts, with computation_strategy samples, inputs_params, extra_log_factors, data and all_platedims.
    siedas = computation_strategy.split_args(
        name=name, 
        sample=sample, 
        inputs_params=inputs_params, 
        extra_log_factors=extra_log_factors, 
        data=data,
        all_platedims=all_platedims,
    )

    lpq_func = _logPQ_plate if computation_strategy is no_checkpoint else _logPQ_plate_checkpointed

    lpq = None
    for sieda in siedas:
        lpq = lpq_func(
            name=name,
            P=P,
            Q=Q,
            scope=scope,
            active_platedims=active_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampler=sampler,
            computation_strategy=computation_strategy,
            **sieda,
            prev_lpq = lpq
        )

    return lpq

def _logPQ_plate_checkpointed(*args, **kwargs):
    return t.utils.checkpoint.checkpoint(_logPQ_plate_args_kwargs, args, kwargs, use_reentrant=False)

def _logPQ_plate_args_kwargs(args, kwargs):
    return _logPQ_plate(*args, **kwargs)

def _logPQ_plate(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample: dict, 
        inputs_params: dict,
        data: dict,
        extra_log_factors: dict, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampler:Sampler,
        computation_strategy:Optional[Split],
        prev_lpq):

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)

    assert isinstance(sample, dict)
    assert isinstance(inputs_params, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)


    #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
    #by name=None.
    if name is not None:
        new_platedim = all_platedims[name]
        active_platedims = [*active_platedims, new_platedim]

    scope = update_scope(scope, Q, sample, inputs_params)

    lps, all_Ks, K_inits, K_prevs, K_currs = lp_getter(
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
        sampler=sampler,
        computation_strategy=computation_strategy)

    #Sum out Ks for non-timeseries variables.
    #Returns a single tensor, with dimensions:
    # Higher plates
    # K_prevs from timeseries
    # K_currs from timeseries
    # Ks from higher plates.
    lp = reduce_Ks(lps, all_Ks)

    #Sum over new_platedim
    if name is not None:
        if 0 < len(K_prevs):
            #Timeseries
            lp = lp.order(new_platedim, K_prevs, K_currs)    # Removes torchdims from T, and Ks
            lp = chain_logmmexp(lp)# Kprevs x Kcurrs
            assert 2 == lp.ndim

            #Backpropagating info.
            if prev_lpq is None:
                #No prev_lpq, so we're either not split or we're on the last split;
                #sum over Kcurr.
                lp = lp.logsumexp(-1)
            else:
                #prev_lpq, so we're split.
                prev_lpq = prev_lpq.order(K_currs)[:, None] #Unnamed dims: Kcurrs x 1
                lp = logmmexp(lp, prev_lpq).squeeze(-1)     #Unnamed dims: Kprevs
            assert 1 == lp.ndim

            #Label unnamed K_inits
            lp = generic_getitem(lp, K_inits)

        else:
            #No timeseries
            lp = lp.sum(new_platedim)

            if prev_lpq is not None:
                assert set(generic_dims(lp)) == set(generic_dims(prev_lpq))
                lp = prev_lpq + lp

    return lp

def logPQ_dist(
        name:str,
        P:Dist, 
        Q:Union[Dist, Data],
        sample: OptionalTensor,
        inputs_params: dict,
        data: OptionalTensor,
        extra_log_factors: None,
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampler:Sampler,
        computation_strategy:Optional[Split]):

    assert isinstance(P, Dist)
    assert isinstance(Q, (Dist, Data))

    assert isinstance(sample, OptionalTensor)
    assert inputs_params is None
    assert isinstance(data, OptionalTensor)
    assert extra_log_factors is None

    #Either sample or data is None.
    if sample is None:
        assert data is not None
        assert isinstance(Q, Data)
        sample_data = data
    else: 
        #sample is not None
        assert data is None
        assert isinstance(Q, Dist)
        sample_data = sample

    lpq = P.log_prob(sample=sample_data, scope=scope)

    if sample is not None:
        Kdim = groupvarname2Kdim[name]
        lq = Q.log_prob(sample=sample, scope=scope)
        lq = sampler.reduce_logQ(lq, active_platedims, Kdim)

        lpq = lpq - lq - math.log(Kdim.size)
        
    return lpq


def logPQ_timeseries(
        name:str,
        P:Group, 
        Q:Group, 
        sample: dict, 
        inputs_params: dict,
        data: None,
        extra_log_factors: None, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampler:Sampler,
        computation_strategy:Optional[Split]):

    assert isinstance(P, Timeseries)
    assert isinstance(Q, (Timeseries, Plate, Data))

    assert isinstance(sample, OptionalTensor)
    assert inputs_params is None
    assert isinstance(data, OptionalTensor)
    assert extra_log_factors is None

    #Either sample or data is None.
    if sample is None:
        assert data is not None
        assert isinstance(Q, Data)
        sample_data = data
    else: 
        #sample is not None
        assert data is None
        assert isinstance(Q, Dist)
        sample_data = sample


    T_dim = active_platedims[-1]
    K_dim = groupvarname2Kdim[name]

    lpq, Kprev_dim = P.log_prob(sample=sample_data, scope=scope, T_dim=T_dim, K_dim=K_dim)

    initial_state = None
    if initial_state is None:
        initial_state = scope[self.init]

    sample_prev = t.cat([
        initial_state,
        sample.order[T_dim][:[-1]],
    ], 0)[T_dim]

    scope = {**scope}
    scope[name] = sample_prev
    self.trans.log_prob(sample=sample_data, scope=scope)

    if sample is not None:
        Kdim = groupvarname2Kdim[name]
        lq = Q.log_prob(sample=sample, scope=scope)
        lq = sampler.reduce_logQ(lq, active_platedims, Kdim) # Think about whether this makes sense for timeseries.

        lpq = lpq - lq - math.log(Kdim.size)
        
    return (lpq, K_dim, K_prev)


def logPQ_group(
        name:str,
        P:Group, 
        Q:Group, 
        sample: dict, 
        inputs_params: dict,
        data: None,
        extra_log_factors: None, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampler:Sampler,
        computation_strategy:Optional[Split]):

    assert isinstance(P, Group)
    assert isinstance(Q, Group)


    assert isinstance(sample, dict)
    assert inputs_params is None
    assert data is None
    assert extra_log_factors is None

    Kdim = groupvarname2Kdim[name]
    all_Kdims = set(groupvarname2Kdim.values())

    total_logP = 0.
    total_logQ = 0.
    for childname, childP in P.prog.items():
        childQ = Q.prog[childname]
        childsample = sample[childname]
        assert isinstance(childP, Dist)
        assert isinstance(childQ, Dist)
        assert isinstance(childsample, Tensor)

        total_logP = total_logP + childP.log_prob(sample=childsample, scope=scope)
        total_logQ = total_logQ + childQ.log_prob(sample=childsample, scope=scope)

    total_logQ = sampler.reduce_logQ(total_logQ, active_platedims, Kdim)

    logPQ = total_logP - total_logQ - math.log(Kdim.size)
    return logPQ


def lp_getter(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample: dict, 
        inputs_params: dict,
        data: dict,
        extra_log_factors: dict, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        sampler:Sampler,
        computation_strategy:Optional[Split]):
    """Traverses Q according to the structure of P collecting log probabilities
    
    """

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)

    #We want to pass back just the incoming scope, as nothing outside the plate can see
    #variables inside the plate.  So `scope` is the internal scope, and `parent_scope`
    #is the external scope we will pass back.

    assert set(P.prog.keys()) == set(Q.prog.keys())

    lps = list(tree_values(extra_log_factors).values())
    K_inits = []
    K_prevs = []
    K_currs = []

    for childname, childP in P.prog.items():
        childQ = Q.prog.get(childname) 

        #childQ doesn't necessarily have a distribution if sample_data is data.
        #childQ defaults to None in that case.

        if isinstance(childP, Dist):
            assert isinstance(childQ, (Dist, Data))
            method = logPQ_dist
        elif isinstance(childP, Timeseries):
            assert isinstance(childQ, (Dist, Timeseries, Data))
            method = logPQ_timeseries
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
            inputs_params=inputs_params.get(childname),
            extra_log_factors=extra_log_factors.get(childname),
            scope=scope, 
            active_platedims=active_platedims,
            all_platedims=all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampler=sampler,
            computation_strategy=computation_strategy)
        if isinstance(lp, tuple):
            lp, K_init, K_prev, K_curr = lp
            K_inits.append(K_init)
            K_prevs.append(K_prev)
            K_currs.append(K_curr)
        
        lps.append(lp)

    #Collect all Ks in the plate
    all_Ks = []
    for varname, dist in Q.prog.items():
        if isinstance(dist, (Dist, Group)):
            all_Ks.append(groupvarname2Kdim[varname])
        else:
            assert isinstance(dist, (Plate, Data, Timeseries))
            
    return lps, all_Ks, K_inits, K_prevs, K_currs
