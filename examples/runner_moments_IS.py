import torch as t
from alan import Split, mean
import pickle
import time
import hydra
import importlib.util
import sys
from pathlib import Path

def safe_time(device):
    if device == 'cuda':
        t.cuda.synchronize()
    return time.time()

@hydra.main(version_base=None, config_path='config', config_name='moments_IS_conf')
def run_experiment(cfg):
    print(cfg)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    method = cfg.method
    assert method in ['mpis', 'global_is']

    num_runs = cfg.num_runs
    warmup_runs = cfg.warmup_runs
    Ks = cfg.Ks
    reparam = cfg.reparam
    N_predll = cfg.N_predll

    split_plate = cfg.split.plate
    split_size = cfg.split.size

    dataset_seed = cfg.dataset_seed
    fake_data = cfg.fake_data

    min_K_for_split = cfg.split.min_K
    if min_K_for_split is None:
        min_K_for_split = 0

    spec = importlib.util.spec_from_file_location(cfg.model, f"{cfg.model}/{cfg.model}.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = model
    spec.loader.exec_module(model)

    # Make sure all the required folders exist for this model
    for folder in ['results/moments', 'job_status/moments', 'plots/moments']:
        Path(f"{cfg.model}/{folder}").mkdir(parents=True, exist_ok=True)

    t.manual_seed(0)
    if not fake_data:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates = model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', False)
    else:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, _ = model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', True, return_fake_latents=True)

    # Put extended data, covariates and (if fake_data==True) fake_latents on device
    for key in all_data:
        all_data[key] = all_data[key].to(device)
    for key in all_covariates:
        all_covariates[key] = all_covariates[key].to(device)
    if fake_data:
        for key in fake_latents:
            fake_latents[key] = fake_latents[key].to(device)

    elbos = t.zeros((len(Ks), num_runs)).to(device)
    p_lls = t.zeros((len(Ks), num_runs)).to(device)

    temp_P = model.get_P(platesizes, covariates)
    latent_names = list(temp_P.varname2groupvarname().keys())
    latent_names.remove('obs')

    moment_list = list(zip(latent_names, [mean]*len(latent_names)))

    # Technically we're only doing MSE when fake_data==True, since we know the true latent values,
    # however, the total biased sample variance estimator is the same as the MSE if we use sample means as ground truths.
    # Then we can rescale by num_samples/(num_samples-1) to get the unbiased sample variance estimator
    MSEs = {name: t.zeros((len(Ks))).to(device) for name in latent_names} 

    times = {"elbos":   t.zeros((len(Ks), num_runs)),
             "moments": t.zeros((len(Ks), num_runs)),
             "p_ll":    t.zeros((len(Ks), num_runs))}

    job_status_file = f"{cfg.model}/job_status/moments/{cfg.method}{'_FAKE_DATA' if fake_data else ''}_status.txt"
    if cfg.write_job_status:
        with open(job_status_file, "w") as f:
            f.write(f"Starting job.\n")

    for K_idx, K in enumerate(Ks):

        print(f"K: {K}")
        K_start_time = safe_time(device)

        moments_collection = {}

        seed = 0

        for num_run in range(-warmup_runs, num_runs):
            num_failed = 0
            failed = True 

            while failed and num_failed < 10:
                try:
                    seed += 1

                    print(num_run)

                    if method != 'global_is' and split_plate is not None and split_size is not None and K >= min_K_for_split:
                        split = Split(split_plate, split_size)
                    else:
                        # assert (split_plate is None and split_size is None) or K < min_K_for_split
                        split = None

                    t.manual_seed(seed)

                    prob = model.generate_problem(device, platesizes, data, covariates, Q_param_type='opt')

                    sample_start_time = safe_time(device)
                    sample = prob.sample(K, reparam) if method == 'mpis' else prob.sample_nonmp(K, reparam)
                    sample_time = safe_time(device) - sample_start_time

                    elbo_start_time = safe_time(device)
                    elbo = sample.elbo_nograd() if split is None else sample.elbo_nograd(computation_strategy = split)
                    elbo_time = safe_time(device) - elbo_start_time

                    moments_start_time = safe_time(device)
                    moments = sample.moments(moment_list)
                    moments_time = safe_time(device) - moments_start_time                

                    p_ll_start_time = safe_time(device)
                    importance_sample = sample.importance_sample(N=N_predll) if split is None else sample.importance_sample(N=N_predll, computation_strategy = split)
                    extended_importance_sample = importance_sample.extend(all_platesizes, all_covariates)
                    ll = extended_importance_sample.predictive_ll(all_data)
                    p_ll_time = safe_time(device) - p_ll_start_time
                    
                    if num_run >= 0:
                        # only save results for the actual runs, not the warmup runs
                        elbos[K_idx, num_run] = elbo.item()

                        for i, name in enumerate(latent_names):
                            if num_run == 0:
                                moments_collection[name] = t.zeros((num_runs, *moments[i].shape), names=(None, *moments[i].names)).to(device)
                            moments_collection[name][num_run] = moments[i]

                        p_lls[K_idx, num_run] = ll['obs'].item()

                        times["elbos"][K_idx, num_run]   = sample_time + elbo_time
                        times["moments"][K_idx, num_run] = sample_time + moments_time
                        times["p_ll"][K_idx, num_run]    = sample_time + p_ll_time
                    
                    failed = False
                
                except ValueError as e:
                    num_failed += 1

                    print(e)
                    if cfg.write_job_status:
                        with open(job_status_file, "a") as f:
                            f.write(f"Error in num_run: {num_run}.\n")
                    continue

            if num_failed >= 10:
                print(f"Failed to complete num_run: {num_run} after 10 attempts (using seeds {seed-num_failed}-{seed}).")
                break

        for i, name in enumerate(latent_names):
            if fake_data:
                ground_truth = fake_latents[name]
                if (None, *ground_truth.names) != moments_collection[name].names:
                    ground_truth = ground_truth.align_as(moments_collection[name]).mean(0)
            else:
                ground_truth = moments_collection[name].mean(0)

            MSEs[name][K_idx] = ((moments_collection[name] - ground_truth)**2).mean(0).sum()

            # if we're using real data, we rescale to obtain the unbiased sample variance estimator
            if not fake_data:
                MSEs[name][K_idx] *= num_runs/(num_runs-1)

        if cfg.write_job_status:
            with open(job_status_file, "a") as f:
                f.write(f"K: {K} done in {safe_time(device)-K_start_time}s.\n")

        to_pickle = {'elbos': elbos.cpu(), 'p_lls': p_lls.cpu(), 'MSEs': {k: v.cpu() for k, v in MSEs.items()},
                    'times': times, 'Ks': Ks,  'num_runs': num_runs}

        with open(f'{cfg.model}/results/moments/{cfg.method}{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
            pickle.dump(to_pickle, f)

        print()

if __name__ == "__main__":
    run_experiment()