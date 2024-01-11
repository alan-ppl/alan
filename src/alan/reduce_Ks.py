import opt_einsum
from .utils import *
from .unravel_index import unravel_index

def einsum_args(lps, sum_dims):
    """
    opt_einsum requires pretty weird arguments to get an optimal path.
    This function constructs the required arguments.
    """
    #There shouldn't be any non-torchdim dimensions.
    #Should eventually be able to implement this as a straight product-sum
    for lp in lps:
        assert lp.shape == ()

    set_sum_dims = set(sum_dims)

    all_dims = unify_dims(lps)
    dim_to_idx = {dim: i for (i, dim) in enumerate(all_dims)}
    out_dims = [dim for dim in all_dims if dim not in set_sum_dims]
    out_idxs = [dim_to_idx[dim] for dim in out_dims]

    undim_lps = []
    arg_idxs = []
    for lp in lps:
        dims = generic_dims(lp)
        arg_idxs.append([dim_to_idx[dim] for dim in dims])
        undim_lps.append(generic_order(lp, dims))

    assert all(not is_dimtensor(lp) for lp in undim_lps)

    return [val for pair in zip(undim_lps, arg_idxs) for val in pair] + [out_idxs], out_dims


def sample_Ks(lps, Ks_to_sum, N_dim, num_samples):

    """
    Fundamental method that returns K samples from the posterior
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
    assert set(unify_dims(lps)).issuperset(Ks_to_sum)
    
    _, lps_for_sampling, Ks_to_sample = collect_lps(lps, Ks_to_sum)

    #Now that we have the list of reduced factors and which Kdims to sample from each factor we can sample from each factor in turn
    indices = {}
    
    for lps, kdims_to_sample in zip(lps_for_sampling[::-1], Ks_to_sample[::-1]): 
        lp = sum(lps)

        for dim in list(set(generic_dims(lp)).intersection(set(indices.keys()))):
            lp = lp.order(dim)[indices[dim]]
        
        #If there is more than one Kdim to sample from this factor we need to sample from the joint distribution
        #To do this we sample from a multinomial over the indices of the lp tensor
        #We then unravel the indices and assign them to the appropriate Kdim

        # shift lps up by the max value in each kdim_to_sample to avoid numerical issues
        lp_max = lp.amax(kdims_to_sample)

        sampled_flat_idx = t.multinomial(t.exp(lp.order(*kdims_to_sample) - lp_max).ravel(), num_samples, replacement=True)
        unravelled_indices = unravel_index(sampled_flat_idx, shape=[dim.size for dim in kdims_to_sample])
        
        for idx, kdim in zip(unravelled_indices, kdims_to_sample):
            indices[kdim] = idx[N_dim]

                
        #Otherwise we can just sample from the multinomial with probabilities given by the Kdim dimension of the lp tensor
        # else:
        #     indices[kdims_to_sample[0]] = t.multinomial(t.exp(lp.order(*kdims_to_sample)), num_samples, replacement=True)[N_dim]

        
    return indices
    
    
def reduce_Ks(lps, Ks_to_sum):
    """
    Sum over Ks_to_sum, returning a single tensor.
    """
    assert_unique_dim_iter(Ks_to_sum)

    result, _, _ = collect_lps(lps, Ks_to_sum)

    return result

def checkpoint_reduce_Ks(lps, Ks_to_sum):
    return t.utils.checkpoint.checkpoint(reduce_Ks, lps, Ks_to_sum, use_reentrant=False)

def logsumexp_sum(_Ks_to_sum, *lps_to_reduce):
    #Needs a strange argument order, because checkpoint doesn't work with lists of lps.
    return logsumexp_dims(sum(lps_to_reduce), _Ks_to_sum, ignore_extra_dims=True)



def collect_lps(lps, Ks_to_sum):
    """
    Helper method that sums over Ks and returns a list of the reduced tensors along with a list of which Ks were reduced over for each reduced tensor.
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
        
    args, out_dims = einsum_args(lps, Ks_to_sum)
    path = opt_einsum.contract_path(*args)[0]
    
    all_reduced_lps = [[*lps]]
    Ks_to_sample = []
    
    for lp_idxs in path:
        #Split lps into two groups: those we're going to reduce, and the rest.
        lps_to_reduce = tuple(lps[i] for i in lp_idxs)
        lps = [lps[i] for i in range(len(lps)) if i not in lp_idxs]

        #In this step, sum over all Ks in Ks_to_sample, and not in lps (i.e. the other tensors)
        _Ks_to_sum = tuple(set(Ks_to_sum).difference(unify_dims(lps)).intersection(unify_dims(lps_to_reduce)))
        Ks_to_sample.append(_Ks_to_sum)

        #Instantiates but doesn't save lp with _Ks_to_sample dims
        lps.append(checkpoint(logsumexp_sum, _Ks_to_sum, *lps_to_reduce, use_reentrant=False))
        #lps.append(logsumexp_sum(_Ks_to_sum, *lps_to_reduce))
        all_reduced_lps.append([*lps])

    all_reduced_lps = all_reduced_lps[:-1]

    assert 1==len(lps)
    result = lps[0]
    
    return result, all_reduced_lps, Ks_to_sample
