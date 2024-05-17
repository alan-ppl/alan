import pickle
import numpy as np

def get_best_results(model_name, validation_iter_number=200, method_names=['qem', 'rws', 'vi', 'qem_nonmp', 'global_vi', 'global_rws'], global_K = 10, dataset_seeds=[0], ignore_nans=True):
    print(f"Getting best results for {model_name}.")

    results = {method_name: {} for method_name in method_names}

    elbos = {method_name: [] for method_name in method_names}
    p_lls = {method_name: [] for method_name in method_names}

    elbo_stderrs = {method_name: [] for method_name in method_names}
    p_ll_stderrs = {method_name: [] for method_name in method_names}

    iter_times = {method_name: [] for method_name in method_names}
    iter_times_stderrs = {method_name: [] for method_name in method_names}

    output_to_pickle = {method_name: {} for method_name in method_names}

    # select whether to ignore NaNs or not
    mean_func = np.nanmean if ignore_nans else np.mean
    std_func = np.nanstd if ignore_nans else np.std

    with open(f'../experiments/results/{model_name}/Ks_lrs.pkl', 'rb') as f:
        Ks_lrs = pickle.load(f)
    # pad Ks_lrs with None for the missing Ks
    max_num_lrs = max([len(lrs) for lrs in Ks_lrs.values()])
    for K in Ks_lrs:
        Ks_lrs[K] = Ks_lrs[K] + [None] * (max_num_lrs - len(Ks_lrs[K]))

    for method_name in method_names:

        # Load the results
        for dataset_seed in dataset_seeds:
            if 'global' not in method_name:
                with open(f'../experiments/results/{model_name}/{method_name}{dataset_seed}.pkl', 'rb') as f:
                    results[method_name][dataset_seed] = pickle.load(f)

                    # Extract the elbos, p_lls and iter_times
                    if method_name != 'HMC':
                        elbos[method_name].append(results[method_name][dataset_seed]['elbos'])
                        iter_times[method_name].append(results[method_name][dataset_seed]['iter_times'])
                    else:
                        iter_times[method_name].append(results[method_name][dataset_seed]['times']['p_ll'])
                    p_lls[method_name].append(results[method_name][dataset_seed]['p_lls'])

            else:
                # this branch loads results from the moments (autodiff) paper 
                with open(f'results/{model_name}/moments/{method_name[7:]}{global_K}K{dataset_seed}.pkl', 'rb') as f:
                    results[method_name][dataset_seed] = pickle.load(f)

                    # add in a Ks key
                    results[method_name][dataset_seed]['Ks'] = [global_K]

                    # Extract the elbos, p_lls and iter_times and pad them by prepending a singleton dimension (for K)
                    elbos[method_name].append(np.expand_dims(results[method_name][dataset_seed]['elbos'], 0))

                    elbos[method_name].append(results[method_name][dataset_seed]['elbos'].expand((1, *results[method_name][dataset_seed]['elbos'].shape)))
                    p_lls[method_name].append(results[method_name][dataset_seed]['p_lls'].expand((1, *results[method_name][dataset_seed]['p_lls'].shape)))
                    iter_times[method_name].append(results[method_name][dataset_seed]['times']['elbos'].expand((1, *results[method_name][dataset_seed]['times']['elbos'].shape)))

        # Check that the learning rates, Ks, num_iters and num_runs are the same for this method throughout all seeds
        for key in ['lrs', 'Ks', 'num_iters', 'num_runs']:
            if not (method_name == 'HMC'):# and key in ['lrs', 'Ks']):
                assert np.all([results[method_name][dataset_seed][key] == results[method_name][dataset_seeds[0]][key] for dataset_seed in dataset_seeds])

        if method_name != 'HMC':
            # Average out over the seeds and convert any 0 elbos or p_lls to NaNs
            elbos[method_name] = np.stack(elbos[method_name])
            p_lls[method_name] = np.stack(p_lls[method_name])
            iter_times[method_name] = np.stack(iter_times[method_name])

            elbos[method_name][elbos[method_name] == 0] = np.nan
            p_lls[method_name][p_lls[method_name] == 0] = np.nan
            iter_times[method_name][iter_times[method_name] == 0] = np.nan

            # Average out over the seed dimension
            elbos[method_name] = mean_func(elbos[method_name], 0)  
            p_lls[method_name] = mean_func(p_lls[method_name], 0)
            iter_times[method_name] = mean_func(iter_times[method_name], 0)
            # N.B. each of these is now a 4D array of shape (num_Ks, num_lrs, num_iters, num_runs)

            # Compute stds over the runs
            elbo_stderrs[method_name] = std_func(elbos[method_name], 3) / np.sqrt(elbos[method_name].shape[3])
            p_ll_stderrs[method_name] = std_func(p_lls[method_name], 3) / np.sqrt(p_lls[method_name].shape[3])
            iter_times_stderrs[method_name] = std_func(iter_times[method_name], 3) / np.sqrt(iter_times[method_name].shape[3])

            # Average out over the run dimension
            elbos[method_name] = mean_func(elbos[method_name], 3) 
            p_lls[method_name] = mean_func(p_lls[method_name], 3)
            iter_times[method_name] = mean_func(iter_times[method_name], 3)
            # N.B. each of these is now a 3D array of shape (num_Ks, num_lrs, num_iters)

            # For each K, order the lr dimensions in order of decreasing elbo at the validation_iter_number
            for k, K in enumerate(results[method_name][dataset_seeds[0]]['Ks']):
                lr_order = np.argsort(elbos[method_name][k, :, validation_iter_number])[::-1]

                # # remove lr idxs corresponding to NaNs
                # lr_order = lr_order[~np.isnan(elbos[method_name][k, lr_order, validation_iter_number])]

                # move lr idxs corresponding to NaNs to the end of lr_order
                lr_order = np.concatenate([lr_order[~np.isnan(elbos[method_name][k, lr_order, validation_iter_number])],
                                        lr_order[np.isnan(elbos[method_name][k, lr_order, validation_iter_number])]])

                results[method_name][dataset_seeds[0]]['lrs'] = Ks_lrs[K]
                lrs = np.array(results[method_name][dataset_seeds[0]]['lrs'])[lr_order]

                print(f"{method_name} K: {K}, lr_order: {lr_order} ({lrs})")

                elbos[method_name][k] = elbos[method_name][k, lr_order, :]
                p_lls[method_name][k] = p_lls[method_name][k, lr_order, :]
                iter_times[method_name][k] = iter_times[method_name][k, lr_order, :]

                elbo_stderrs[method_name][k] = elbo_stderrs[method_name][k, lr_order, :]
                p_ll_stderrs[method_name][k] = p_ll_stderrs[method_name][k, lr_order, :]
                iter_times_stderrs[method_name][k] = iter_times_stderrs[method_name][k, lr_order, :]

                # Save the results
                output_to_pickle[method_name][K] = {'elbos': elbos[method_name][k], 
                                                    'p_lls': p_lls[method_name][k], 
                                                    'iter_times': iter_times[method_name][k], 
                                                    'elbo_stderrs': elbo_stderrs[method_name][k],
                                                    'p_ll_stderrs': p_ll_stderrs[method_name][k],
                                                    'iter_times_stderrs': iter_times_stderrs[method_name][k],
                                                    'lrs': lrs}
        else:
            # Average out over the seeds and convert any 0 elbos or p_lls to NaNs
            p_lls[method_name] = np.stack(p_lls[method_name])
            iter_times[method_name] = np.stack(iter_times[method_name])

            p_lls[method_name][p_lls[method_name] == 0] = np.nan
            iter_times[method_name][iter_times[method_name] == 0] = np.nan

            # Average out over the seed dimension
            p_lls[method_name] = mean_func(p_lls[method_name], 0)
            iter_times[method_name] = mean_func(iter_times[method_name], 0)
            # N.B. each of these is now a 2D array of shape (num_iters, num_runs)

            # Compute stds over the runs
            p_ll_stderrs[method_name] = std_func(p_lls[method_name], -1) / np.sqrt(p_lls[method_name].shape[-1])
            iter_times_stderrs[method_name] = std_func(iter_times[method_name], -1) / np.sqrt(iter_times[method_name].shape[-1])

            # Average out over the run dimension
            p_lls[method_name] = mean_func(p_lls[method_name], -1)
            iter_times[method_name] = mean_func(iter_times[method_name], -1)
            output_to_pickle[method_name] = {'elbos': np.nan, 
                                             'p_lls': p_lls[method_name], 
                                             'iter_times': iter_times[method_name], 
                                             'elbo_stderrs': np.nan,
                                             'p_ll_stderrs': p_ll_stderrs[method_name],
                                             'iter_times_stderrs': iter_times_stderrs[method_name],
                                             'lrs': []}
            
    # Save the results
    with open(f'../experiments/results/{model_name}/best.pkl', 'wb') as f:
        pickle.dump(output_to_pickle, f)

    print()

if __name__ == "__main__":
    method_names = ['qem', 'rws', 'vi', 'qem_nonmp']


    get_best_results('bus_breakdown', method_names=method_names)
    get_best_results('bus_breakdown_reparam', method_names=method_names)

    # get_best_results('chimpanzees', method_names=method_names)

    get_best_results('movielens', method_names=method_names)
    get_best_results('movielens_reparam', method_names=method_names)

    get_best_results('occupancy', method_names=['qem', 'rws', 'qem_nonmp'])

    get_best_results('radon', method_names=['qem', 'rws', 'qem_nonmp'])
    
    get_best_results('covid', method_names=method_names)