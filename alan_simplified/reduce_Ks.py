import opt_einsum
from .utils import *

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


def sample_Ks(lps, Ks_to_sum, num_samples=1):
    """
    Fundamental method that returns K samples from the posterior
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
    print(lps)
    print(Ks_to_sum)
    assert set(unify_dims(lps)).issuperset(Ks_to_sum)
    
    args, out_dims = einsum_args(lps, Ks_to_sum)
    path = opt_einsum.contract_path(*args)[0]

    N_dim = Dim('N')
    
    lps_for_sampling = [lps.copy()]
    Ks_to_sample = []
    
    for lp_idxs in path[:-1]:
        #Split lps into two groups: those we're going to reduce, and the rest.
        lps_to_reduce = tuple(lps[i] for i in lp_idxs)
        lps = [lps[i] for i in range(len(lps)) if i not in lp_idxs]

        #In this step, sum over all Ks in Ks_to_sample, and not in lps (i.e. the other tensors)
        _Ks_to_sum = tuple(set(Ks_to_sum).difference(unify_dims(lps)))
        Ks_to_sample.append(_Ks_to_sum)

        #Instantiates but doesn't save lp with _Ks_to_sample dims
        lps.append(checkpoint(logsumexp_sum, _Ks_to_sum, *lps_to_reduce, use_reentrant=False))
        lps_for_sampling.append(lps.copy())

    Ks_to_sample.append(tuple(set(Ks_to_sum).intersection(unify_dims(lps_for_sampling[-1]))))


    #Now that we have the list of reduced factors and which Kdims to sample from each factor we can sample from each factor in turn
    indices = {}
    sampled_Ks = []
    for lps, kdims_to_sample in zip(lps_for_sampling[::-1], Ks_to_sample[::-1]): 
        lp = sum(lps)

        for dim in list(set(generic_dims(lp)).intersection(sampled_Ks)):
            lp = lp.order(dim)[indices[str(dim)]]

        #If there is more than one Kdim to sample from this factor we need to sample from the joint distribution
        #To do this we sample from a multinomial over the indices of the lp tensor
        #We then unravel the indices and assign them to the appropriate Kdim
        if len(kdims_to_sample) > 1:
            for idx, kdim in zip(t.unravel_index(t.multinomial(t.exp(lp.order(*kdims_to_sample)).ravel(), num_samples, replacement=True), shape=[dim.size for dim in kdims_to_sample]), kdims_to_sample):
                indices[str(kdim)] = idx[N_dim]
                sampled_Ks.append(kdim)
                
        #Otherwise we can just sample from the multinomial with probabilities given by the Kdim dimension of the lp tensor
        else:
            indices[str(kdims_to_sample[0])] = t.multinomial(t.exp(lp.order(*kdims_to_sample)), num_samples, replacement=True)[N_dim]
            sampled_Ks.append(kdims_to_sample[0])


        sampled_Ks.append(kdims_to_sample)
            

    #Remove N_dim from indices that was only used for indexing into subsequent factors
    for k,v in indices.items():
        indices[k] = v.order(N_dim)
        
    return indices
    
    
def reduce_Ks(lps, Ks_to_sum):
    """
    Fundamental method that sums over Ks
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)

    args, out_dims = einsum_args(lps, Ks_to_sum)
    path = opt_einsum.contract_path(*args)[0]

    for lp_idxs in path:
        #Split lps into two groups: those we're going to reduce, and the rest.
        lps_to_reduce = tuple(lps[i] for i in lp_idxs)
        lps = [lps[i] for i in range(len(lps)) if i not in lp_idxs]

        #In this step, sum over all Ks in Ks_to_sum, and not in lps (i.e. the other tensors)
        _Ks_to_sum = tuple(set(Ks_to_sum).difference(unify_dims(lps)))

        #Instantiates but doesn't save lp with _Ks_to_sum dims
        lps.append(checkpoint(logsumexp_sum, _Ks_to_sum, *lps_to_reduce, use_reentrant=False))
        #lps.append(logsumexp_sum(_Ks_to_sum, *lps_to_reduce))

    assert 1==len(lps)
    result = lps[0]

    return result

def logsumexp_sum(_Ks_to_sum, *lps_to_reduce):
    #Needs a strange argument order, because checkpoint doesn't work with lists of lps.
    return logsumexp_dims(sum(lps_to_reduce), _Ks_to_sum, ignore_extra_dims=True)

