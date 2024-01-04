import math
from typing import Optional, Union

from .Plate import Plate, tree_values, update_scope
from .Group import Group
from .Data import Data
from .Timeseries import Timeseries
from .dist import Dist, datagroup

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

    scope = update_scope(scope, inputs_params)
    scope = update_scope(scope, sample)

    lps, all_Ks, K_inits, K_currs = lp_getter(
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
        if 0 < len(K_inits):
            #Timeseries
            lp = lp.order(new_platedim, K_inits, K_currs)    # Removes torchdims from T, and Ks
            lp = chain_logmmexp(lp)# Kprevs x Kcurrs
            assert 2 == lp.ndim

            lp = lp.logsumexp(-1)
            assert 1 == lp.ndim
            #Put torchdim back. 
            #Stupid trailing None is necessary, because otherwise the list of K_inits is just splatted in, rather than being treated as a group.
            lp = lp[K_inits, None].squeeze(-1)

            assert prev_lpq is None

            #Backpropagating info.
            #if prev_lpq is None:
            #    #No prev_lpq, so we're either not split or we're on the last split;
            #    #sum over Kcurr.
            #    lp = lp.logsumexp(-1)
            #else:
            #    #prev_lpq, so we're split.
            #    prev_lpq = prev_lpq.order(K_currs)[:, None] #Unnamed dims: Kcurrs x 1
            #    lp = logmmexp(lp, prev_lpq).squeeze(-1)     #Unnamed dims: Kprevs
            #assert 1 == lp.ndim

        else:
            #No timeseries
            lp = lp.sum(new_platedim)

            if prev_lpq is not None:
                assert set(generic_dims(lp)) == set(generic_dims(prev_lpq))
                lp = prev_lpq + lp

    return lp



def logPQ_timeseries(
        name:str,
        P:Timeseries,
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
    assert isinstance(Q, (Timeseries, Dist, Data))

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
        assert isinstance(Q, (Dist, Timeseries))
        sample_data = sample


    T_dim     = active_platedims[-1]
    K_dim     = groupvarname2Kdim[name]
    Kinit_dim = groupvarname2Kdim[P.init]

    initial_state = scope[P.init]
    sample_prev = sample.order(K_dim)[Kinit_dim]

    sample_prev = t.cat([
        initial_state[None, ...],
        sample.order(T_dim)[:-1],
    ], 0)[T_dim]

    scope = {**scope}
    scope[name] = sample_prev

    lpq = P.trans.log_prob(sample=sample_data, scope=scope)

    if sample is not None:
        Kdim = groupvarname2Kdim[name]
        lq = Q.log_prob(sample=sample, scope=scope)
        lq = sampler.reduce_logQ(lq, active_platedims, Kdim) # Think about whether this makes sense for timeseries.

        lpq = lpq - lq - math.log(Kdim.size)
        
    return lpq, Kinit_dim, K_dim

def logPQ_gdt(
        name:str,
        P:dict,
        Q:dict,
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

    assert isinstance(sample, dict)
    assert inputs_params is None
    assert extra_log_factors is None
    assert isinstance(P, dict)
    assert isinstance(Q, dict)

    prog_P = P
    prog_Q = Q

    assert 1<=len(prog_P) 
    assert set(prog_P.keys()) == set(prog_Q.keys())

    #Immediately return if data.
    if datagroup(prog_Q):
        assert len(prog_Q) == 1
        k = next(iter(prog_Q))

        assert isinstance(prog_Q[k], Data)
        assert sample[k] is None
        assert isinstance(data[k], Tensor)

        return prog_P[k].log_prob(sample=data[k], scope=scope)

    Kdim = groupvarname2Kdim[name]
    total_logP = 0.
    total_logQ = 0.

    for k in prog_P:
        dist_P = prog_P[k]
        dist_Q = prog_Q[k]
        samp   = sample[k]

        assert isinstance(dist_P, (Dist, Timeseries))
        assert isinstance(dist_Q, (Dist, Timeseries))
        assert isinstance(samp, Tensor)
        assert data[k] is None

        total_logP = total_logP + dist_P.log_prob(sample=samp, scope=scope)
        total_logQ = total_logQ + dist_Q.log_prob(sample=samp, scope=scope)

    total_logQ = sampler.reduce_logQ(total_logQ, active_platedims, Kdim)
    return total_logP - total_logQ - math.log(Kdim.size)

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

    assert set(P.flat_prog.keys()) == set(Q.flat_prog.keys())

    lps = list(tree_values(extra_log_factors).values())
    K_inits = []
    K_currs = []

    for childname, childQ in Q.grouped_prog.items():
        if isinstance(childQ, dict):
            childP = {varname: P.flat_prog[varname] for varname in childQ}
            method = logPQ_gdt
        else:
            assert isinstance(childQ, Plate)
            childP = P.flat_prog[childname]
            assert isinstance(childP, Plate)
            method = logPQ_plate

        lp = method(
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
            sampler=sampler,
            computation_strategy=computation_strategy)
        if isinstance(lp, tuple):
            lp, K_init, K_curr = lp
            K_inits.append(K_init)
            K_currs.append(K_curr)
        
        lps.append(lp)

    #Collect all non-timeseries Ks in the plate (note: iterating through Q!!)
    all_Ks = []
    for varname, dist in Q.grouped_prog.items():
        if isinstance(dist, dict):
            if not datagroup(dist):
                all_Ks.append(groupvarname2Kdim[varname])
        else:
            assert isinstance(dist, Plate)
            
    return lps, all_Ks, K_inits, K_currs
