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

def sample_logPQ(
    P: Plate,
    Q: Plate,
    sample: dict,
    inputs_params_P: dict,
    inputs_params_Q: dict,
    data: dict,
    extra_log_factors: dict,
    scope_P: dict[str, Tensor],
    scope_Q: dict[str, Tensor],
    active_platedims: list[Dim],
    all_platedims: dict[str:Dim],
    groupvarname2Kdim: dict[str, Tensor],
    sampling_type: SamplingType,
    split: Optional[Split],
    number_of_samples: int,
):
    
    assert isinstance(sample, dict)
    assert isinstance(inputs_params_P, dict)
    assert isinstance(inputs_params_Q, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)
    
    indices = {}
    
    
    #Assume we have P samples,and Q samples both in the form of a dict of tensors with the same keys
    #and with a nested plate structure
    P_samples = {}
    Q_samples = {}
    
    #We also have log probs in same format?
    P_log_probs = {}
    Q_log_probs = {}
    
    for plate_idx in range(len(all_platedims)):
        plate = all_platedims[plate_idx]
        #Will need this to determine which lps to sum
        lower_plate_dims = all_platedims[plate_idx+1:]
        
        #Maybe we want the factors seperately? Currently summing together
        current_log_pq_factor = sum([lp - lq for lp, lq in zip(P_log_probs[plate], Q_log_probs[plate])])
        
        
        #We want to index into samples somewhere here
        
        
        
        #Sum over lower plates and Ks
        summed_log_pqs = []
        for lower_plate in lower_plate_dims[::-1]:
            log_pqs = [lp - lq for lp, lq in zip(P_log_probs[lower_plate], Q_log_probs[lower_plate])] + summed_log_pqs
            summed_log_pqs.append(reduce_Ks(log_pqs, lower_plate))
        
        
        #At the end should just have the logpqs in the current plate and a factor which has all Ks and plates (from lower plates) summed out
        assert len(summed_log_pqs) == 1
        summed_log_pq = summed_log_pqs[0]
        
        current_log_pq_factor = current_log_pq_factor + summed_log_pq
        
        #Now we need to sample from the current plate
        indices.update(sample_Ks(current_log_pq_factor, [plate], number_of_samples))
        
        

        
    #return indices?
    return indices

