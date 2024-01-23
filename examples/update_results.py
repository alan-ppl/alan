import pickle
import numpy as np
from typing import List

def combine_lrs(model: str, results_subfolders: List[str], ouput_folder: str, methods: List[str] = ['qem', 'rws', 'vi'], dataset_seed: int = 0, Ks_to_keep='all'):
    ouput = {}
    for method in methods:
        ouput[method] = {'lrs': []}
        for folder in results_subfolders:
            with open(f'{model}/results/{folder}/{method}{dataset_seed}.pkl', 'rb') as f:
                result = pickle.load(f)

                if 'Ks' not in ouput[method]:
                    if Ks_to_keep == 'all':
                        Ks_to_keep = result['Ks']

                K_idx_to_keep = []
                for K_idx, K in enumerate(result['Ks']):
                    if K in Ks_to_keep:
                        K_idx_to_keep.append(K_idx)

                assert len(K_idx_to_keep) == len(Ks_to_keep)
                
                ouput[method]['Ks'] = Ks_to_keep

                for key in ['num_runs', 'num_iters']:
                    if key in ouput[method]:
                        assert np.all(ouput[method][key] == result[key])
                    else:
                        ouput[method][key] = result[key]

                for lr in result['lrs']:
                    if lr not in ouput[method]['lrs']:
                        ouput[method]['lrs'] = ouput[method]['lrs'] + result['lrs']

                for key in ['elbos', 'p_lls', 'iter_times']:
                    if key not in ouput[method]:
                        ouput[method][key] = result[key][K_idx_to_keep, :, :, :]
                    else:
                        ouput[method][key] = np.concatenate([ouput[method][key], result[key][K_idx_to_keep, :, :, :]], axis=1)

    for method in methods:
        # before writing to disk, sort by lr

        lr_idxs = np.argsort(ouput[method]['lrs'])
        ouput[method]['lrs'] = np.array(ouput[method]['lrs'])[lr_idxs]
        for key in ['elbos', 'p_lls', 'iter_times']:
            ouput[method][key] = ouput[method][key][:, lr_idxs, :, :]

        with open(f'{model}/results/{ouput_folder}/{method}{dataset_seed}.pkl', 'wb') as f:
            pickle.dump(ouput[method], f)

if __name__ == '__main__':
    combine_lrs('chimpanzees', ['K5_15', 'K5_15_lr0.3'], 'K5_15_lr_all', Ks_to_keep=[5,15])