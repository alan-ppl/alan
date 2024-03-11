import torch as t
from alan import Split, mean, mean2
import pickle
import time
import hydra
import importlib.util
import sys
from pathlib import Path

from itertools import product

from torch.distributions import NegativeBinomial, Poisson

def safe_time(device):
    if device == 'cuda':
        t.cuda.synchronize()
    return time.time()

@hydra.main(version_base=None, config_path='config', config_name='conf_covid')
def run_experiment(cfg):
    print(cfg)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    Ks = cfg.Ks
    lrs = cfg.lrs
    num_runs = cfg.num_runs
    num_iters = cfg.num_iters

    do_predll = cfg.predll.do_predll
    N_predll = cfg.predll.N_predll

    reparam = cfg.reparam

    split_plate = cfg.split.plate
    split_size = cfg.split.size

    if split_plate is not None and split_size is not None:
        split = Split(split_plate, split_size)
    else:
        assert split_plate is None and split_size is None
        split = None

    spec = importlib.util.spec_from_file_location(cfg.model, f"covid/{cfg.model}.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = model
    spec.loader.exec_module(model)

    # Make sure all the required folders exist for this model
    for folder in ['results', 'job_status', 'plots']:
        Path(f"covid/{folder}/{cfg.model}").mkdir(parents=True, exist_ok=True)

    platesizes, all_platesizes, data, all_data, covariates, all_covariates = model.load_data_covariates(device, cfg.dataset_seed, 'covid/data/')

    # Put extended data and covariates on device
    for key in all_data:
        all_data[key] = all_data[key].to(device)
    for key in all_covariates:
        all_covariates[key] = all_covariates[key].to(device)

    elbos = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs)).to(device)
    p_lls = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs)).to(device)

    # the below times should NOT include predictive ll computation time, as this is optional
    iter_times = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs))

    if cfg.write_job_status:
        with open(f"covid/job_status/{cfg.model}/{cfg.method}_status.txt", "w") as f:
            f.write(f"Starting job.\n")

    for num_run in range(num_runs):
        for K_idx, K in enumerate(Ks):
            K_start_time = safe_time(device)
            for lr_idx, lr in enumerate(lrs):
                t.manual_seed(num_run)
                print(f"K: {K}, lr: {lr}, num_run: {num_run}")

                prob = model.generate_problem(device, platesizes, data, covariates, Q_param_type='qem' if cfg.method == 'qem' else 'opt')

                if cfg.method == 'vi': 
                    opt = t.optim.Adam(prob.Q.parameters(), lr=lr)
                elif cfg.method == 'rws':
                    opt = t.optim.Adam(prob.Q.parameters(), lr=lr, maximize=True)
                
                try:
                    for i in range(num_iters+1):
                        elbo_start_time = safe_time(device)

                        if cfg.method == 'vi' or cfg.method == 'rws':
                            opt.zero_grad()

                        sample = prob.sample(K, reparam)

                        if cfg.method == 'vi':
                            elbo = sample.elbo_vi() if split is None else sample.elbo_vi(computation_strategy = split)
                        elif cfg.method == 'rws':
                            elbo = sample.elbo_rws() if split is None else sample.elbo_rws(computation_strategy = split)
                        elif cfg.method == 'qem':
                            elbo = sample.elbo_nograd() if split is None else sample.elbo_nograd(computation_strategy = split)

                        elbo_end_time = safe_time(device)

                        elbos[K_idx, lr_idx, i, num_run] = elbo.item()

                        if do_predll:
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
                        
                        t.save(prob.state_dict(), f"covid/results/{cfg.model}/{cfg.method}_{cfg.dataset_seed}_{K}_{lr}.pth")
                        
                        moment_fns = [mean, mean2]
                        if cfg.model in ['covid', 'covid_poisson', 'covid_cn', 'covid_poisson_cn']:
                            names = ['CM_mean', 'CM_ex2', 'Wearing_mean', 'Wearing_ex2', 'Mobility_mean', 'Mobility_ex2']
                            latents = ['CM_alpha', 'Wearing_alpha', 'Mobility_alpha']
                        elif cfg.model in ['poisson_only_wearing_mobility', 'poisson_only_wearing_mobility_cn']:
                            names = ['Mobility_mean', 'Mobility_ex2', 'Wearing_mean', 'Wearing_ex2']
                            latents = ['Wearing_alpha', 'Mobility_alpha']
                        elif cfg.model in ['poisson_only_npis', 'poisson_only_npis_cn']:
                            names = ['CM_mean', 'CM_ex2']
                            latents = ['CM_alpha']

                        moments = {name:[] for name in names}
                        for _ in range(100):
                            sample = prob.sample(K, reparam)
                            
                            #Moments of the weights
                            desired_moments = tuple(product(latents, moment_fns))
                            
                            m = sample.moments(desired_moments)
                            for j, k in enumerate(moments.keys()):
                                moments[k].append(m[j].rename(None))
                                
                            #convert moments to numpy
                        
                        for k,v in moments.items():
                            moments[k] = t.stack(v).detach().mean(0).cpu().numpy()
                            
                                
                        #save moments to file
                        with open(f'covid/results/{cfg.model}/{cfg.method}_moments_{K}_{lr}.pkl', 'wb') as f:
                            pickle.dump(moments, f)
                            
                        # predictive samples
                        names = ['log_infected', 'log_infected_ex2']
                        desired_moments = (('log_infected', mean), ('log_infected', mean2))

                        moments = dict(zip(names, sample.moments(desired_moments)))

                        log_infected = moments['log_infected'].rename(None)

                        if cfg.model in ['covid', 'covid_cn']:
                            predicted_obs = {'obs':NegativeBinomial(total_count = 1000, probs=1000/(1000 + t.exp(log_infected) + 1e-4)).sample(t.Size([100]))}
                        if cfg.model in ['covid_poisson', 'poisson_only_wearing_mobility', 'poisson_only_npis', 'covid_poisson_cn', 'poisson_only_wearing_mobility_cn', 'poisson_only_npis_cn']:
                            predicted_obs = {'obs':Poisson(rate = t.exp(log_infected) + 1e-6).sample(t.Size([100]))}
                            
                            
                        #convert to numpy
                        predicted_obs['obs'] = predicted_obs['obs'].detach().cpu().numpy()

                        #save predictive samples to file
                        with open(f'covid/results/{cfg.model}/{cfg.method}_predictive_samples_{K}_{lr}.pkl', 'wb') as f:
                            pickle.dump(predicted_obs, f)
                        
                except Exception as e:
                    print(f"num_run: {num_run} K: {K} lr: {lr} failed at iteration {i} with exception {e}.")
                    if cfg.write_job_status:
                        with open(f"covid/job_status/{cfg.model}/{cfg.method}_status.txt", "a") as f:
                            f.write(f"num_run: {num_run} K: {K} lr: {lr} failed at iteration {i} with exception {e}.\n")
                    continue

                
            
            if cfg.write_job_status:
                with open(f"covid/job_status/{cfg.model}/{cfg.method}_status.txt", "a") as f:
                    f.write(f"num_run: {num_run} K: {K} done in {safe_time(device)-K_start_time}s.\n")




    
    
    to_pickle = {'elbos': elbos.cpu(), 'p_lls': p_lls.cpu(), 'iter_times': iter_times,
                 'Ks': Ks, 'lrs': lrs, 'num_runs': num_runs, 'num_iters': num_iters}

    

    for K_idx, K in enumerate(Ks):
        for lr_idx, lr in enumerate(lrs):
            print(f"K: {K}, lr: {lr}")
            print(f"elbo: {elbos[K_idx, lr_idx, 0,:].mean():.3f}")
            print(f"p_ll: {p_lls[K_idx, lr_idx, 0,:].mean():.3f}")
            print()

    # breakpoint()
    with open(f'covid/results/{cfg.model}/{cfg.method}{cfg.dataset_seed}.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)


if __name__ == "__main__":
    run_experiment()