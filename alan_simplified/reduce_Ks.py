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

