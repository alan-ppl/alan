import pickle
import numpy as np
from typing import List

def combine_lrs(model: str, results_subfolders: List[str], output_folder: str, methods: List[str] = ['qem', 'rws', 'vi'], dataset_seed: int = 0, Ks_to_keep='all'):
    output = {}
    for method in methods:
        output[method] = {'lrs': []}
        for folder in results_subfolders:
            with open(f'{model}/results/{folder}/{method}{dataset_seed}.pkl', 'rb') as f:
                result = pickle.load(f)

                if 'Ks' not in output[method]:
                    if Ks_to_keep == 'all':
                        Ks_to_keep = result['Ks']

                K_idx_to_keep = []
                for K_idx, K in enumerate(result['Ks']):
                    if K in Ks_to_keep:
                        K_idx_to_keep.append(K_idx)

                assert len(K_idx_to_keep) == len(Ks_to_keep)
                
                output[method]['Ks'] = Ks_to_keep

                for key in ['num_runs', 'num_iters']:
                    if key in output[method]:
                        assert np.all(output[method][key] == result[key])
                    else:
                        output[method][key] = result[key]

                for lr in result['lrs']:
                    if lr not in output[method]['lrs']:
                        output[method]['lrs'] = output[method]['lrs'] + result['lrs']

                for key in ['elbos', 'p_lls', 'iter_times']:
                    if key not in output[method]:
                        output[method][key] = result[key][K_idx_to_keep, :, :, :]
                    else:
                        output[method][key] = np.concatenate([output[method][key], result[key][K_idx_to_keep, :, :, :]], axis=1)

    for method in methods:
        # before writing to disk, sort by lr

        lr_idxs = np.argsort(output[method]['lrs'])[::-1]
        output[method]['lrs'] = np.array(output[method]['lrs'])[lr_idxs]
        for key in ['elbos', 'p_lls', 'iter_times']:
            output[method][key] = output[method][key][:, lr_idxs, :, :]

        with open(f'{model}/results/{output_folder}/{method}{dataset_seed}.pkl', 'wb') as f:
            pickle.dump(output[method], f)

def combine_Ks(model: str, results_subfolders: List[str], output_folder: str, methods: List[str] = ['qem', 'rws', 'vi'], dataset_seed: int = 0):
    output = {}
    for method in methods:
        output[method] = {'Ks': []}

        results = []
        for folder in results_subfolders:
            with open(f'{model}/results/{folder}/{method}{dataset_seed}.pkl', 'rb') as f:
                result = pickle.load(f)
                results.append(result)

        assert all([set(result['lrs']) == set(results[0]['lrs']) for result in results])
        assert all([result['num_runs'] == results[0]['num_runs'] for result in results])
        assert all([result['num_iters'] == results[0]['num_iters'] for result in results])

        output[method]['num_runs'] = results[0]['num_runs']
        output[method]['num_iters'] = results[0]['num_iters']

        Ks = []
        for result in results:
            assert set(result['Ks']).intersection(set(Ks)) == set()
            Ks += result['Ks']
        output[method]['Ks'] = Ks

        # order lrs
        lr_idxs = np.argsort(results[0]['lrs'])[::-1].tolist()
        lrs = np.array(results[0]['lrs'])[lr_idxs].tolist()
        output[method]['lrs'] = lrs

        # for each result's elbo, p_ll, iter_times, sort by lr
        for result in results:
            for key in ['elbos', 'p_lls', 'iter_times']:
                result[key] = result[key][:, lr_idxs, :, :]

        # concatenate elbos, p_lls, iter_times across results along K dimension
        for result in results:
            for key in ['elbos', 'p_lls', 'iter_times']:
                if key not in output[method]:
                    output[method][key] = result[key]
                else:
                    output[method][key] = np.concatenate([output[method][key], result[key]], axis=0)


    for method in methods:
        # before writing to disk, sort by K

        K_idxs = np.argsort(output[method]['Ks']).tolist()
        output[method]['Ks'] = np.array(output[method]['Ks'])[K_idxs].tolist()
        for key in ['elbos', 'p_lls', 'iter_times']:
            output[method][key] = output[method][key][K_idxs, :, :, :]

        with open(f'{model}/results/{output_folder}/{method}{dataset_seed}.pkl', 'wb') as f:
            pickle.dump(output[method], f)

if __name__ == '__main__':
    # combine_lrs('bus_breakdown', ['final0.1-0.0001', 'lr0.3-1'], 'final1-0.0001', Ks_to_keep=[3,10,30], methods=['qem'])

    # combine_lrs('chimpanzees', ['K5_15', 'K5_15_lr0.3'], 'K5_15_lr_0.001-0.3', Ks_to_keep=[5,15])
    # combine_lrs('chimpanzees', ['K5_15', 'K5_15_lr0.3', 'K5_15_lr_0.5-1'], 'K5_15_lr_0.001-1', Ks_to_keep=[5,15])

    # combine_lrs('occupancy', ['lr0.01-0.1-0.3', 'lr0.5-1'], 'lr0.01-1', Ks_to_keep=[3,5,10], methods=['qem'])    
    
    # combine_lrs('movielens', ['regular_version_final_FULL', 'regular_version_lr_0.5-1'], 'regular_version_final_FULL_all_lrs', Ks_to_keep=[3,10,30], methods=['qem'])

    # combine_Ks('chimpanzees', ['K5_15_lr_0.001-1', 'K10_lr_0.001-1'], 'K5_10_15_lr_0.001-1')