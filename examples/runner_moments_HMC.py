# import torch as t
import numpy as np
import pickle
import time
import hydra
import importlib.util
import sys
from pathlib import Path

import pymc as pm
# import pymc3 as pm

import pymc.sampling.jax as pmjax
import jax

import logging
logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)

def safe_time(device):
    # if device == 'cuda':
        # t.cuda.synchronize()
    return time.time()

@hydra.main(version_base=None, config_path='config', config_name='moments_HMC_conf')
def run_experiment(cfg):
    print(cfg)

    # device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    device = 'cpu'
    print("torch device: ", device)

    print("pymc device: ", jax.default_backend())  # should print 'gpu'

    num_runs = cfg.num_runs
    dataset_seed = cfg.dataset_seed
    fake_data = cfg.fake_data

    num_samples = cfg.num_samples
    num_tuning_samples = cfg.num_tuning_samples
    target_accept = cfg.target_accept

    spec = importlib.util.spec_from_file_location(cfg.model, f"{cfg.model}/{cfg.model}.py")
    alan_model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = alan_model
    spec.loader.exec_module(alan_model)

    spec = importlib.util.spec_from_file_location(cfg.model, f"HMC/{cfg.model}/{cfg.model}.py")
    pymc_model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = pymc_model
    spec.loader.exec_module(pymc_model)

    # Make sure all the required folders exist for this model
    for folder in ['results/moments', 'job_status/moments', 'plots/moments']:
        Path(f"{cfg.model}/{folder}").mkdir(parents=True, exist_ok=True)

    if not fake_data:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates = alan_model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', False)
    else:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, _ = alan_model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', True, return_fake_latents=True)

    temp_P = alan_model.get_P(platesizes, covariates)
    latent_names = list(temp_P.varname2groupvarname().keys())
    latent_names.remove('obs')

    var_names_to_track = latent_names + ['obs', 'p_ll']

    # Put extended data, covariates and (if fake_data==True) fake_latents on device
    # and convert to numpy for pymc
    for data_covs in [data, all_data, covariates, all_covariates]:
        for key in data_covs:
            data_covs[key] = data_covs[key].numpy()
    if fake_data:
        for key in fake_latents:
            fake_latents[key] = fake_latents[key].numpy()

    p_lls = np.zeros((num_samples, num_runs))

    # Technically we're only doing MSE when fake_data==True, since we know the true latent values,
    # however, the total biased sample variance estimator is the same as the MSE if we use sample means as ground truths.
    # Then we can rescale by num_samples/(num_samples-1) to get the unbiased sample variance estimator
    MSEs = {name: np.zeros((num_samples)) for name in latent_names} 

    times = {"moments": np.zeros((num_samples, num_runs)),
             "p_ll":   np.zeros((num_samples, num_runs))}

    job_status_file = f"{cfg.model}/job_status/moments/HMC{'_FAKE_DATA' if fake_data else ''}_status.txt"
    if cfg.write_job_status:
        with open(job_status_file, "w") as f:
            f.write(f"Starting job.\n")

    moments_collection = {}

    for num_run in range(num_runs):
        if num_run==1:
            continue

        print(f"num_run: {num_run}")
        num_run_start_time = safe_time(device)

        model = pymc_model.get_model(data, covariates)
        with model:
            p_ll = pm.Deterministic('p_ll', model.observedlogp)

            print("Sampling posterior with JAX")
            trace = pmjax.sample_blackjax_nuts(draws=num_samples, tune=num_tuning_samples, chains=1, random_seed=num_run, target_accept=target_accept)
            times['moments'][:, num_run] = np.linspace(0,trace.attrs["sampling_time"],num_samples+num_tuning_samples+1)[num_tuning_samples+1:]
            
            # compute moments for each latent
            for name in latent_names:
                if num_run == 0:
                    latent_shape = trace.posterior[name].mean(("chain", "draw")).shape
                    moments_collection[name] = np.zeros((num_samples, num_runs, *latent_shape))
                moments_collection[name][:, num_run, ...] = np.array([trace.posterior[name][:,:j].mean(("chain", "draw")).data for j in range(1, num_samples+1)])

            # do predictive log likelihood
            pm.set_data(pymc_model.get_test_data_cov_dict(all_data, all_covariates, platesizes))

            print("Sampling predictive log likelihood with JAX")
            p_ll_start_time = safe_time(device)
            pp_trace = pm.sample_posterior_predictive(trace, var_names=var_names_to_track, random_seed=num_run, predictions=True, progressbar=True)#, return_inferencedata=True)
            print(f"p_ll sampling time: {safe_time(device)-p_ll_start_time}s")
            test_ll = pp_trace.predictions.p_ll.mean('chain').data
            print("extracted test_ll")
            times['p_ll'][:, num_run] = np.linspace(0,safe_time(device)-p_ll_start_time,num_samples+1)[1:] + times['moments'][:, num_run]
            
            p_lls[:, num_run] = test_ll

        if cfg.write_job_status:
            with open(job_status_file, "a") as f:
                f.write(f"Done num_run: {num_run} in {safe_time(device)-num_run_start_time}s.\n")

    breakpoint()
    for i, name in enumerate(latent_names):
        if fake_data:
            ground_truth = fake_latents[name]
            # if (None, *ground_truth.names) != moments_collection[name].names:
            #     ground_truth = ground_truth.align_as(moments_collection[name]).mean(0)
        else:
            ground_truth = moments_collection[name].mean(0)

        MSEs[name] = ((moments_collection[name] - ground_truth)**2).mean(0).sum()

        # if we're using real data, we rescale to obtain the unbiased sample variance estimator
        if not fake_data:
            MSEs[name] *= num_runs/(num_runs-1)

    to_pickle = {'p_lls': p_lls, 'MSEs': MSEs,
                'times': times, 'num_runs': num_runs, 
                'num_samples': num_samples, 'num_tuning_samples': num_tuning_samples, 'target_accept': target_accept}

    with open(f'{cfg.model}/results/moments/HMC{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)

    print()

if __name__ == "__main__":
    run_experiment()