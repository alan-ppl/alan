import torch as t
from alan import Split, mean, mean2
import pickle
import time
import hydra
import importlib.util
import sys
from pathlib import Path

from itertools import product

import numpy as np
def safe_time(device):
    if device.type == 'cuda':
        t.cuda.synchronize()
    return time.time()

@hydra.main(version_base=None, config_path='config', config_name='conf')
def run_experiment(cfg):
    start_time = time.asctime()
    print(start_time)
    print(cfg)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print("Device:", device)

    Ks_lrs = cfg.Ks_lrs

    Ks = [K for K, lrs in Ks_lrs.items()]

        
    #longest list of lrs:
    num_lrs = max([len(lrs) for K, lrs in Ks_lrs.items()])
    
    num_runs = cfg.num_runs
    num_iters = cfg.num_iters

    do_predll = cfg.predll.do_predll
    N_predll = cfg.predll.N_predll

    reparam = cfg.reparam

    split_plate = cfg.split.plate
    split_size = cfg.split.size
    
    if cfg.model_name == '':
        model_name = cfg.model
    else:
        model_name = cfg.model_name

    if split_plate is not None and split_size is not None:
        split = Split(split_plate, split_size)
    else:
        assert split_plate is None and split_size is None
        split = None


    non_mp_string = '_nonmp' if cfg.non_mp else ''

    spec = importlib.util.spec_from_file_location(cfg.model, f"{cfg.model}/{model_name}.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = model
    spec.loader.exec_module(model)

    # Make sure all the required folders exist for this model
    for folder in ['results', 'job_status', 'plots', 'moments']:
        Path(f"experiments/{folder}/{model_name}").mkdir(parents=True, exist_ok=True)
        
    

    with open(f"experiments/results/{model_name}/Ks_lrs.pkl", "wb") as f:
        pickle.dump(Ks_lrs, f)
        
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = model.load_data_covariates(device, cfg.dataset_seed, f'{cfg.model}/data/')

    # Put extended data and covariates on device
    for key in all_data:
        all_data[key] = all_data[key].to(device)
    for key in all_covariates:
        all_covariates[key] = all_covariates[key].to(device)

    elbos = t.zeros((len(Ks), num_lrs, num_iters+1, num_runs)).to(device)
    p_lls = t.zeros((len(Ks), num_lrs, num_iters+1, num_runs)).to(device)


    temp_P = model.get_P(platesizes, covariates)
    latent_names = list(temp_P.varname2groupvarname().keys())
    latent_names.remove('obs')

    moment_list = list(product(latent_names, [mean, mean2]))
    
    # the below times should NOT include predictive ll computation time, as this is optional
    iter_times = t.zeros((len(Ks), num_lrs, num_iters+1, num_runs))

    if cfg.write_job_status:
        with open(f"experiments/job_status/{model_name}/{cfg.method}{non_mp_string}_status.txt", "w") as f:
            f.write(f"{start_time}\n{cfg}\n\n")
            f.write(f"Starting job.\n")

    moments_collection = {'means':{}, 'means2':{}}
    
    
    for K_idx, K in enumerate(Ks):
        lrs = Ks_lrs[K]
        K_start_time = safe_time(device)
        for lr_idx, lr in enumerate(lrs):
            for num_run in range(num_runs):
                t.manual_seed(num_run)
                print(f"K: {K}, lr: {lr}, num_run: {num_run}")

                prob = model.generate_problem(device, platesizes, data, covariates, Q_param_type='qem' if cfg.method == 'qem' else 'opt')

                moments_collection = {'means':{}, 'means2':{}}
                sample = prob.sample(3, reparam=False)
                m = sample.moments(moment_list)
                temp_means = [m[i] for i in range(0,len(latent_names)*2,2)]
                for j, k in enumerate(latent_names):
                    latent_shape = temp_means[j].shape
                    moments_collection['means'][k] = np.zeros((num_iters+1, num_runs, *latent_shape))
                    moments_collection['means2'][k] = np.zeros((num_iters+1, num_runs, *latent_shape))
        
                if cfg.method == 'vi': 
                    opt = t.optim.Adam(prob.Q.parameters(), lr=lr)
                elif cfg.method == 'rws':
                    opt = t.optim.Adam(prob.Q.parameters(), lr=lr, maximize=True)
                
                try:
                    for i in range(num_iters+1):
                        elbo_start_time = safe_time(device)

                        if cfg.method == 'vi' or cfg.method == 'rws':
                            opt.zero_grad()

                        if cfg.non_mp: 
                            sample = prob.sample_nonmp(K, reparam)
                        else:
                            sample = prob.sample(K, reparam)

                        if cfg.non_mp:
                            #Non-mp doesn't take computation_strategy
                            if cfg.method == 'vi':
                                elbo = sample.elbo_vi()
                            elif cfg.method == 'rws':
                                elbo = sample.elbo_rws()
                            elif cfg.method == 'qem':
                                elbo = sample.elbo_nograd()
                        else:
                            if cfg.method == 'vi':
                                elbo = sample.elbo_vi() if split is None else sample.elbo_vi(computation_strategy = split)
                            elif cfg.method == 'rws':
                                elbo = sample.elbo_rws() if split is None else sample.elbo_rws(computation_strategy = split)
                            elif cfg.method == 'qem':
                                elbo = sample.elbo_nograd() if split is None else sample.elbo_nograd(computation_strategy = split)                            

                        elbo_end_time = safe_time(device)

                        elbos[K_idx, lr_idx, i, num_run] = elbo.item()

                        if do_predll:
                            if cfg.non_mp:
                                importance_sample = sample.importance_sample(N=N_predll)
                            else:
                                importance_sample = sample.importance_sample(N=N_predll) if split is None else sample.importance_sample(N=N_predll, computation_strategy = split)
                            extended_importance_sample = importance_sample.extend(all_platesizes, all_covariates)
                            ll = extended_importance_sample.predictive_ll(all_data)
                            
                            p_lls[K_idx, lr_idx, i, num_run] = ll['obs'].item()

                            if i % 50 == 0:
                                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")

                        if not do_predll and i % 50 == 0:
                            print(f"Iter {i}. Elbo: {elbo:.3f}")

                        update_start_time = safe_time(device)

                        if i < num_iters and (cfg.method == 'vi' or cfg.method == 'rws'):
                            (-elbo).backward()
                            opt.step()
                        else:
                            sample.update_qem_params(lr)

                        update_end_time = safe_time(device)

                        iter_times[K_idx, lr_idx, i, num_run] = elbo_end_time - elbo_start_time + update_end_time - update_start_time
                
                        t.save(prob.state_dict(), f"results/{model_name}/{cfg.method}_{cfg.dataset_seed}_{K}_{lr}{non_mp_string}.pth")
                        
                        if cfg.save_moments:
                            
                            sample = prob.sample(K, reparam=False)
                            
                            
                            m = sample.moments(moment_list)
                            temp_means = [m[i] for i in range(0,len(latent_names)*2,2)]
                            temp_means2 = [m[i] for i in range(1,len(latent_names)*2,2)]
                            for j, k in enumerate(latent_names):
                                moments_collection['means'][k][i, num_run, ...] = temp_means[j].cpu().numpy()
                                moments_collection['means2'][k][i, num_run, ...] = temp_means2[j].cpu().numpy()
                    
                    
                except Exception as e:
                    print(f"num_run: {num_run} K: {K} lr: {lr} failed at iteration {i} with exception {e}.")
                    if cfg.write_job_status:
                        with open(f"experiments/job_status/{cfg.model}/{model_name}/{cfg.method}{non_mp_string}_status.txt", "a") as f:
                            f.write(f"num_run: {num_run} K: {K} lr: {lr} failed at iteration {i} with exception {e}.\n")
                    continue
                
            
            #Average over runs
            for j, k in enumerate(latent_names):
                moments_collection['means'][k] = moments_collection['means'][k].mean(axis=1)
                moments_collection['means2'][k] = moments_collection['means2'][k].mean(axis=1)
                
            #save moments to file
            with open(f"experiments/moments/{model_name}/{cfg.method}_{cfg.dataset_seed}_{K}_{lr}_moments{non_mp_string}.pkl", "wb") as f:
                pickle.dump(moments_collection, f)

        if cfg.write_job_status:
            with open(f"experiments/job_status/{model_name}/{cfg.method}{non_mp_string}_status.txt", "a") as f:
                f.write(f"num_run: {num_run} K: {K} done in {safe_time(device)-K_start_time}s.\n")

    to_pickle = {'elbos': elbos.cpu(), 'p_lls': p_lls.cpu(), 'iter_times': iter_times,
                 'Ks': Ks, 'lrs': lrs, 'num_runs': num_runs, 'num_iters': num_iters}

    print()

    for K_idx, K in enumerate(Ks):
        for lr_idx, lr in enumerate(lrs):
            print(f"K: {K}, lr: {lr}")
            print(f"elbo: {elbos[K_idx, lr_idx, 0,:].mean():.3f}")
            print(f"p_ll: {p_lls[K_idx, lr_idx, 0,:].mean():.3f}")
            print()

    if device.type == 'cuda':
        cuda_mem_summary = f"CUDA memory - Card size: {t.cuda.get_device_properties(device).total_memory/(1024**3):.2f}GB, Max allocated: {t.cuda.max_memory_allocated(device)/(1024**3):.2f}GB, Max reserved: {t.cuda.max_memory_reserved(device)/(1024**3):.2f}GB"
        print(cuda_mem_summary)

        if cfg.write_job_status:
            with open(f"experiments/job_status/{model_name}/{cfg.method}{non_mp_string}_status.txt", "a") as f:
                f.write(f"\n{cuda_mem_summary}\n")

    # breakpoint()
    with open(f'experiments/results/{model_name}/{cfg.method}{non_mp_string}{cfg.dataset_seed}.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)


if __name__ == "__main__":
    run_experiment()