import opt_einsum
from .utils import *
from .unravel_index import unravel_index
from functorch.dim import Dim

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

def sample_Ks_timeseries(lps, Ks_to_sum, ts_init_Ks, N_dim, num_samples, T_dim, indices):

    """
    Fundamental method that returns K samples from the posterior *where Ks_to_sum are timeseries K_dims*
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
    assert set(unify_dims(lps)).issuperset(Ks_to_sum)
    # breakpoint()
    _, lps_for_sampling, Ks_to_sample = collect_lps(lps, Ks_to_sum)

    #Now that we have the list of reduced factors and which Kdims to sample from each factor we can sample from each factor in turn
    indices = {**indices}
    # breakpoint()
    
    for lps, kdims_to_sample, init_K_dim in zip(lps_for_sampling[::-1], Ks_to_sample[::-1], ts_init_Ks[::-1]):         
        assert len(kdims_to_sample) == 1
        K_dim = kdims_to_sample[0]

        lp = sum(lps)


        assert K_dim in set(generic_dims(lp))
        assert T_dim in set(generic_dims(lp))
        assert init_K_dim in set(generic_dims(lp))

        assert init_K_dim in indices.keys()

        # index into lp EXCEPT for ts_init_Ks
        for dim in list(set(generic_dims(lp)).intersection(set(indices.keys())).difference(set(ts_init_Ks))):
            lp = lp.order(dim)[indices[dim]]

        #get plate_dims
        plate_dims = list(set(generic_dims(lp)).difference(set(indices.keys()) | set(ts_init_Ks)).difference(set([N_dim, K_dim, T_dim])))
        plate_dim_sizes = [dim.size for dim in plate_dims]

        ts_indices = t.zeros((num_samples, *plate_dim_sizes, T_dim.size, ), dtype=t.int64)[N_dim, plate_dims]


        filtered_t_plus_one = None
        smoothed_t_plus_one = None

        for t_idx in range(T_dim.size-1,-1,-1):
            # print(t_idx, generic_dims(lp), generic_dims(filtered_t_plus_one), generic_dims(smoothed_t_plus_one))
            
            # this filtering/forward-run gives us log p(x_t | y_{1:t}, x_{1:t-1}) as a K x K tensor 
            filtered_t = chain_logmmexp(lp.order(T_dim, init_K_dim, K_dim)[:t_idx+1])[init_K_dim, K_dim]
            
            # index into filtered_t with the ts_init_Ks indices to get the filtering distribution as a K-long vector
            filtered_t = filtered_t.order(init_K_dim)[indices[init_K_dim]]
            filtered_t = filtered_t.order(N_dim)
            filtered_t = t.logsumexp(filtered_t, 0) #- t.tensor(num_samples).log().to(filtered_t.device)

            # normalise
            filtered_t = filtered_t - t.logsumexp(filtered_t.order(K_dim), 0)

            # now do smoothing/backward-run to get log p(x_t | y_{1:T}, x_{1:T})
            # this is calculated as 
            #       p(x_t | y_{1:t}, x_{1:t-1}) * INTEGRATE{dx_{t+1} * p(x_{t+1} | y_{1:T}) * p(x_{t+1} | x_t) / p(x_{t+1} | y_{1:t}) }
            # i.e.  filtered_t * INTEGRATE{dx_{t+1} * smoothed_{t+1} * p(x_{t+1} | x_t) / filtered_{t+1} }
            # 
            # see e.g. http://www.gatsby.ucl.ac.uk/~byron/nlds/briers04.pdf
            if t_idx < T_dim.size-1:
                # transition = lp.order(T_dim)[t_idx:t_idx+2]
                # transition = t.cat([filtered_t.expand((1,)), filtered_t_plus_one.expand((1,))], dim=0)
                transition = lp.order(T_dim)[t_idx+1]
                # breakpoint()

                # transition = transition.order(init_K_dim)[indices[init_K_dim]]
                # transition = transition.order(N_dim)
                # transition = t.logsumexp(transition, 0)

                integrand = ((smoothed_t_plus_one - filtered_t_plus_one) + transition) # [2, init_K_dim, K_dim] with torchdims
                # integrand = integrand.order(init_K_dim, K_dim).transpose(-1,0) # [init_K_dim, K_dim, 2] without torchdims
                # integrand = integrand.transpose(-1,-2)                         # [2, init_K_dim, K_dim] without torchdims (which is what we want for chain_logmmexp)

                # breakpoint()
                
                # smoothed_t = filtered_t + chain_logmmexp(integrand)[init_K_dim, K_dim]
                # smoothed_t = filtered_t + t.logsumexp(chain_logmmexp(integrand), -1)[init_K_dim]
                # smoothed_t = filtered_t + t.logsumexp(chain_logmmexp(integrand), 0)[K_dim]

                smoothed_t = filtered_t + integrand.logsumexp(K_dim)

                # index into smoothed_t with the ts_init_Ks indices to get the smoothed distribution as a K-long vector
                smoothed_t = smoothed_t.order(init_K_dim)[indices[init_K_dim]]
                smoothed_t = smoothed_t.order(N_dim)
                smoothed_t = t.logsumexp(smoothed_t, 0)

                # integrand = integrand.order(init_K_dim)[indices[init_K_dim]]
                # smoothed_t = filtered_t + integrand.logsumexp(N_dim)

                # smoothed_t = filtered_t + integrand.logsumexp(init_K_dim)

                # normalise
                smoothed_t = smoothed_t - t.logsumexp(smoothed_t.order(K_dim), 0)

            else:
                smoothed_t = filtered_t
                
                # index into filtered_t with the ts_init_Ks indices to get the filtering distribution as a K-long vector
                # filtered_t = filtered_t.order(init_K_dim)[indices[init_K_dim]]
                # filtered_t = filtered_t.order(N_dim)
                # filtered_t = t.logsumexp(filtered_t, 0)



            # save for next iteration
            filtered_t_plus_one = filtered_t
            smoothed_t_plus_one = smoothed_t

            # index into smoothed_t with the ts_init_Ks indices to get the smoothed distribution as a K-long vector
            # smoothed_t = smoothed_t.order(init_K_dim)[indices[init_K_dim]]
            # smoothed_t = smoothed_t.order(N_dim)
            # smoothed_t = t.logsumexp(smoothed_t, 0)

            # do the same for filtered_t
            # filtered_t = filtered_t.order(init_K_dim)[indices[init_K_dim]]
            # filtered_t = filtered_t.order(N_dim)
            # filtered_t = t.logsumexp(filtered_t, 0)

            # save for next iteration
            # filtered_t_plus_one = filtered_t
            # smoothed_t_plus_one = smoothed_t

            # print(smoothed_t)
            # shift lps up by the max value in each kdim_to_sample to avoid numerical issues
            lp_max = smoothed_t.amax(kdims_to_sample)
            
            # breakpoint()
            # sampled_flat_idx = t.multinomial(t.exp(smoothed_t.order(K_dim) - lp_max).ravel(), 1, replacement=True)[0]
            # ts_indices[t_idx] = sampled_flat_idx#[N_dim]

            sampled_flat_idx = t.multinomial(t.exp(smoothed_t.order(K_dim) - lp_max).ravel(), num_samples, replacement=True)
            # print(ts_indices[t_idx])
            # print(sampled_flat_idx[N_dim])
            ts_indices[t_idx] = sampled_flat_idx[N_dim]

            # print(ts_indices)

        # breakpoint()
        # print([ts_indices.order(N_dim)[:,i].unique().shape[0] for i in range(T_dim.size)])

        indices[K_dim] = ts_indices[T_dim] # TODO: try just the final timestep (as we were doing before)
        
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
        lps.append(logsumexp_sum(_Ks_to_sum, *lps_to_reduce))
        all_reduced_lps.append([*lps])

    all_reduced_lps = all_reduced_lps[:-1]

    assert 1==len(lps)
    result = lps[0]

    # Find indices of any empty K sets
    empty_K_idxs = []
    for i in range(len(Ks_to_sample)):
        if Ks_to_sample[i] == ():
            empty_K_idxs.append(i)

    # Remove empty K sets and corresponding reduced lps
    all_reduced_lps = [lps for i, lps in enumerate(all_reduced_lps) if i not in empty_K_idxs]
    Ks_to_sample = [Ks for i, Ks in enumerate(Ks_to_sample) if i not in empty_K_idxs]
    
    return result, all_reduced_lps, Ks_to_sample
