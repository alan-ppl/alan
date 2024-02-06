import torch as t
from alan import Split, mean
import pickle
import time
import hydra
import importlib.util
import sys
from pathlib import Path

K = 1

def safe_time(device):
    if device == 'cuda':
        t.cuda.synchronize()
    return time.time()

@hydra.main(version_base=None, config_path='config', config_name='moments_iterative_conf')
def run_experiment(cfg):
    print(cfg)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    method = cfg.method
    assert method in ['vi', 'rws']

    lrs = cfg.lrs
    num_runs = cfg.num_runs
    num_iters = cfg.num_iters
    reparam = cfg.reparam
    N_predll = cfg.N_predll

    dataset_seed = cfg.dataset_seed
    fake_data = cfg.fake_data

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

    elbos = t.zeros((len(lrs), num_iters+1, num_runs)).to(device)
    p_lls = t.zeros((len(lrs), num_iters+1, num_runs)).to(device)

    temp_P = model.get_P(platesizes, covariates)
    latent_names = list(temp_P.varname2groupvarname().keys())
    latent_names.remove('obs')

    moment_list = list(zip(latent_names, [mean]*len(latent_names)))

    # Technically we're only doing MSE when fake_data==True, since we know the true latent values,
    # however, the total biased sample variance estimator is the same as the MSE if we use sample means as ground truths.
    # Then we can rescale by num_samples/(num_samples-1) to get the unbiased sample variance estimator
    MSEs = {name: t.zeros((len(lrs), num_iters+1)).to(device) for name in latent_names} 

    times = {"elbos":   t.zeros((len(lrs), num_iters+1, num_runs)),
             "moments": t.zeros((len(lrs), num_iters+1, num_runs)),
             "p_ll":    t.zeros((len(lrs), num_iters+1, num_runs))}

    job_status_file = f"{cfg.model}/job_status/moments/{cfg.method}{'_FAKE_DATA' if fake_data else ''}_status.txt"
    if cfg.write_job_status:
        with open(job_status_file, "w") as f:
            f.write(f"Starting job.\n")

    for lr_idx, lr in enumerate(lrs):

        print(f"lr: {lr}")
        lr_start_time = safe_time(device)

        moments_collection = {}

        for num_run in range(num_runs):
            t.manual_seed(num_run)

            prob = model.generate_problem(device, platesizes, data, covariates, Q_param_type='opt')

            if cfg.method == 'vi': 
                opt = t.optim.Adam(prob.Q.parameters(), lr=lr)
            elif cfg.method == 'rws':
                opt = t.optim.Adam(prob.Q.parameters(), lr=lr, maximize=True)
            
            try:
                prev_update_time = 0

                for iter in range(num_iters+1):
                    opt.zero_grad()

                    sample_start_time = safe_time(device)
                    sample = prob.sample(K, reparam)
                    sample_time = safe_time(device) - sample_start_time

                    elbo_start_time = safe_time(device)
                    if method == 'vi':
                        elbo = sample.elbo_vi()
                    elif method == 'rws':
                        elbo = sample.elbo_rws()
                    elbo_time = safe_time(device) - elbo_start_time

                    elbos[lr_idx, iter, num_run] = elbo.item()

                    moments_start_time = safe_time(device)
                    moments = sample.moments(moment_list)
                    moments_time = safe_time(device) - moments_start_time

                    for i, name in enumerate(latent_names):
                        if num_run == 0 and iter == 0:
                            moments_collection[name] = t.zeros((num_iters+1, num_runs, *moments[i].shape), names=(None, None, *moments[i].names)).to(device)
                        moments_collection[name][iter, num_run, ...] = moments[i]

                    p_ll_start_time = safe_time(device)
                    importance_sample = sample.importance_sample(N=N_predll)
                    extended_importance_sample = importance_sample.extend(all_platesizes, all_covariates)
                    ll = extended_importance_sample.predictive_ll(all_data)
                    p_ll_time = safe_time(device) - p_ll_start_time
                    
                    p_lls[lr_idx, iter, num_run] = ll['obs'].item()

                    times["elbos"][lr_idx, iter, num_run]   = prev_update_time + sample_time + elbo_time
                    times["moments"][lr_idx, iter, num_run] = prev_update_time + sample_time + moments_time
                    times["p_ll"][lr_idx, iter, num_run]    = prev_update_time + sample_time + p_ll_time

                    update_start_time = safe_time(device)
                    if i < num_iters:
                        (-elbo).backward()
                        opt.step()
                    update_time = safe_time(device) - update_start_time

                    prev_update_time = update_time
            
            except Exception as e:
                print(f"lr: {lr} num_run: {num_run} failed at iteration {iter} with exception {e}.")
                if cfg.write_job_status:
                    with open(job_status_file, "a") as f:
                        f.write(f"lr: {lr} num_run {num_run} failed at iteration {iter} with exception {e}.\n")
                continue

        for i, name in enumerate(latent_names):
            if fake_data:
                ground_truth = fake_latents[name]
                latent_ndim = ground_truth.ndim # no need for -1 since we haven't yet added the iteration dimension

                if (None, None, *ground_truth.names) != moments_collection[name].names:
                    ground_truth = ground_truth.align_as(moments_collection[name]).mean(1)
                    latent_ndim = ground_truth.ndim - 1
            else:
                ground_truth = moments_collection[name].mean(1)
                latent_ndim = ground_truth.ndim - 1
            
            # below we transpose the moments_collection to have the num_runs dimension first (so that we can subtract the ground_truth)
            MSE = ((moments_collection[name].transpose(1,0) - ground_truth)**2).mean(0)
            
            if latent_ndim > 0:
                MSE = MSE.sum([-(i+1) for i in range(latent_ndim)])

            MSEs[name][lr_idx, :] = MSE

            # if we're using real data, we rescale to obtain the unbiased sample variance estimator
            if not fake_data:
                MSEs[name][lr_idx, :] *= num_runs/(num_runs-1)

        if cfg.write_job_status:
            with open(job_status_file, "a") as f:
                f.write(f"lr: {lr} done in {safe_time(device)-lr_start_time}s.\n")

        print()

    to_pickle = {'elbos': elbos.cpu(), 'p_lls': p_lls.cpu(), 'MSEs': {k: v.cpu() for k, v in MSEs.items()},
                'times': times, 'lrs': lrs, 'num_runs': num_runs, 'num_iters': num_iters}

    with open(f'{cfg.model}/results/moments/{cfg.method}{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)


if __name__ == "__main__":
    run_experiment()