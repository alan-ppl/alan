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
        varname2groupvarname:dict[str, str],
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
            varname2groupvarname=varname2groupvarname,
            sampler=sampler,
            computation_strategy=computation_strategy,
            **sieda,
            prev_lpq = lpq
        )
    assert isinstance(lpq, Tensor)

    return lpq, (), (), ()

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
        varname2groupvarname:dict[str, str],
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

    lps, all_Ks, K_currs, K_inits = lp_getter(
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
        computation_strategy=computation_strategy
    )

    assert len(K_currs) == len(K_inits)

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
            ##Timeseries
            lp = lp.order(new_platedim, K_inits, K_currs)    # Removes torchdims from T, and Ks
            lp = chain_logmmexp(lp)# Kprevs x Kcurrs
            assert 2 == lp.ndim

            lp = lp.logsumexp(-1)
            assert 1 == lp.ndim
            #Put torchdim back. 
            #Stupid trailing None is necessary, because otherwise the list of K_inits is just splatted in, rather than being treated as a group.
            lp = lp[K_inits, None].squeeze(-1)

            assert prev_lpq is None

        else:
            #No timeseries
            lp = lp.sum(new_platedim)

            if prev_lpq is not None:
                assert set(generic_dims(lp)) == set(generic_dims(prev_lpq))
                lp = prev_lpq + lp

    return lp

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
        varname2groupvarname:dict[str, str],
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

        lp, _ = prog_P[k].log_prob(data[k], scope, None, None)

        return lp, (), (), ()

    Kdim = groupvarname2Kdim[name]
    total_logP = 0.
    total_logQ = 0.

    #Gather extra dimensions for timeseries
    Kinit_dims = []
    for v in prog_P.values():
        if isinstance(v, Timeseries):
            Kinit_dims.append(groupvarname2Kdim[varname2groupvarname[v.init]])

    T_dim = active_platedims[-1] if 1<=len(active_platedims) else None

    Kinits = []
    for k in prog_P:
        dist_P   = prog_P[k]
        dist_Q   = prog_Q[k]
        sample_k = sample[k]

        assert isinstance(dist_P, (Dist, Timeseries))
        assert isinstance(dist_Q, (Dist, Timeseries))
        assert isinstance(sample_k, Tensor)
        assert data[k] is None

        lp, Kinit_p = dist_P.log_prob(sample_k, scope=scope, T_dim=T_dim, K_dim=Kdim)
        lq, Kinit_q = dist_Q.log_prob(sample_k, scope=scope, T_dim=T_dim, K_dim=Kdim)

        if Kinit_q is not None:
            assert Kinit_p is Kinit_q

        if Kinit_p is not None:
            assert isinstance(Kinit_p, Dim)
            Kinits.append(Kinit_p)

        total_logP = total_logP + lp
        total_logQ = total_logQ + lq

    total_logQ = sampler.reduce_logQ(total_logQ, active_platedims, Kdim)
    lp = total_logP - total_logQ - math.log(Kdim.size)

    if 1 <= len(Kinits):
        #There's at least one timeseries in the group.
        Kinit0 = Kinits[0]
        for Kid in Kinit_dims:
            assert Kid is Kinit0
        Knon_timeseries = ()
        Ktimeseries = (Kdim,)
        Kinits = (Kinit0,)
    else:
        #No timeseries in the group.
        Knon_timeseries = (Kdim,)
        Ktimeseries = ()
        Kinit = ()
        
    return lp, Knon_timeseries, Ktimeseries, Kinits


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
        varname2groupvarname:dict[str, str],
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
    Knon_timeseries = []
    Ktimeseries = []
    Kinits = []

    for childname, childQ in Q.grouped_prog.items():
        if isinstance(childQ, dict):
            childP = {varname: P.flat_prog[varname] for varname in childQ}
            method = logPQ_gdt
        else:
            assert isinstance(childQ, Plate)
            childP = P.flat_prog[childname]
            assert isinstance(childP, Plate)
            method = logPQ_plate

        lp, _Knon_timeseries, _Ktimeseries, _Kinits = method(
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
            computation_strategy=computation_strategy
        )

        lps.append(lp)
        Knon_timeseries.extend(_Knon_timeseries)
        Ktimeseries.extend(_Ktimeseries)
        Kinits.extend(_Kinits)
        

    #Collect all non-timeseries Ks in the plate (note: iterating through Q!!)
    #all_Ks = []
    #for varname, dist in Q.grouped_prog.items():
    #    if isinstance(dist, dict):
    #        if not datagroup(dist):
    #            all_Ks.append(groupvarname2Kdim[varname])
    #    else:
    #        assert isinstance(dist, Plate)
            
    return lps, Knon_timeseries, Ktimeseries, Kinits
