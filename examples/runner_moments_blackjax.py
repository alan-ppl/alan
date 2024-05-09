# import torch as t
import numpy as np
import pickle
import time
import hydra
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import blackjax

from collections import namedtuple

import logging
logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)

def safe_time(device):
    # if device == 'cuda':
        # t.cuda.synchronize()
    return time.time()

def get_predll(model, samples, rng_key):
    samples = samples._asdict()
    names = list(samples.keys())
    probs = np.zeros(samples[names[0]].shape[0])
    for i in range(samples[names[0]].shape[0]):
        temp_samples = {}
        for name in names:
            temp_samples[name] = np.array(samples[name][i,...])
        temp_samples = namedtuple("model_params", temp_samples.keys())(*temp_samples.values())
        probs[i] = model(temp_samples)
    # vectorized_apply = jax.vmap(model, in_axes=0, out_axes=0)
    # probs = vectorized_apply(samples)


    return probs
                
@hydra.main(version_base=None, config_path='config', config_name='moments_blackjax_conf')
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

    spec = importlib.util.spec_from_file_location(cfg.model, f"blackjax/{cfg.model}/{cfg.model}.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = model
    spec.loader.exec_module(model)

    # Make sure all the required folders exist for this model
    for folder in ['results', 'job_status', 'plots', 'moments']:
        Path(f"experiments/{folder}/{cfg.model}").mkdir(parents=True, exist_ok=True)
    # t.manual_seed(0)
    if not fake_data:
        # platesizes, all_platesizes, data, all_data, covariates, all_covariates = alan_model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', False)
        with open(f'blackjax/{cfg.model}/data/real_data.pkl', 'rb') as f:
            platesizes, all_platesizes, data, test_data, covariates, test_covariates, latent_names = pickle.load(f)
    else:
        with open(f'blackjax/{cfg.model}/data/fake_data.pkl', 'rb') as f:
            platesizes, all_platesizes, data, test_data, covariates, test_covariates, fake_latents, latent_names = pickle.load(f)
        # platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, _ = alan_model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', True, return_fake_latents=True)


    p_lls = np.zeros((num_samples, num_runs))
    times = {"moments": np.zeros((num_samples, num_runs)),
             "p_ll":   np.zeros((num_samples, num_runs))}

    job_status_file = f"experiments/job_status/{cfg.model}/blackjax{'_FAKE_DATA' if fake_data else ''}_status.txt"
    if cfg.write_job_status:
        with open(job_status_file, "w") as f:
            f.write(f"Starting job.\n")

    moments_collection = {}

    seed = 0

    for num_run in range(num_runs):
        num_failed = 0
        failed = True 

        while failed and num_failed < 10:
            # try:
            seed += 1

            if not fake_data:
                with open(f'blackjax/{cfg.model}/data/real_data.pkl', 'rb') as f:
                    platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
            else:
                with open(f'blackjax/{cfg.model}/data/fake_data.pkl', 'rb') as f:
                    platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, latent_names = pickle.load(f)

            print(f"num_run: {num_run}")
            num_run_start_time = safe_time(device)
            joint_logdensity, params, init_param_fn, transform_non_cent_to_cent = model.get_model(data, covariates)
            training_joint_logdensity = lambda params: joint_logdensity(params, data['obs'], covariates)
            
            warmup = blackjax.window_adaptation(blackjax.nuts, training_joint_logdensity)
            # we use 4 chains for sampling
            rng_key = jax.random.key(num_run)
            rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
            init_params = init_param_fn(init_key)

            
            (state, parameters), _ = warmup.run(warmup_key, init_params, num_tuning_samples)

            def inference_loop(rng_key, kernel, initial_state, num_samples):
                @jax.jit
                def one_step(state, rng_key):
                    state, _ = kernel(rng_key, state)
                    return state, state

                keys = jax.random.split(rng_key, num_samples)
                _, states = jax.lax.scan(one_step, initial_state, keys)

                return states
            
            sampling_start_time = safe_time(device)
            rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
            kernel = blackjax.nuts(training_joint_logdensity, **parameters).step
            states = inference_loop(sample_key, kernel, state, num_samples)
            times['moments'][:, num_run] = np.linspace(0,safe_time(device)-sampling_start_time,num_samples+1)[1:]

            states_dict = states.position._asdict()
            #HMC means
            # HMC_means = {key: np.mean(states_dict[key], axis=0) for key in states_dict}

            if cfg.do_predll:
                print("Sampling predictive log likelihood with JAX")
                p_ll_start_time = safe_time(device)
                test_data, test_covariates = model.get_test_data_cov_dict(all_data, all_covariates, platesizes)
                test_joint_logdensity = lambda params: joint_logdensity(params, test_data['true_obs'], test_covariates)
                pred_ll = get_predll(test_joint_logdensity, states.position, rng_key)
                print(f"p_ll sampling time: {safe_time(device)-p_ll_start_time}s")
                p_lls[:, num_run] = pred_ll
                times['p_ll'][:, num_run] = np.linspace(0,safe_time(device)-p_ll_start_time,num_samples+1)[1:] + times['moments'][:, num_run]
            # compute moments for each latent
            
            states_dict = transform_non_cent_to_cent(states_dict, covariates)
            for name in states_dict.keys():
                if num_run == 0:
                    latent_shape = states_dict[name].shape[1:]
                    moments_collection[name] = np.zeros((num_samples, num_runs, *latent_shape))
                
                moments_collection[name][:, num_run, ...] = np.array([states_dict[name][j,...] for j in range(1, num_samples+1)])
                    
            if cfg.write_job_status:
                with open(job_status_file, "a") as f:
                    f.write(f"Done num_run: {num_run} in {safe_time(device)-num_run_start_time}s.\n")

            failed = False

        # except ValueError as e:
        #     num_failed += 1
            
        #     print(e)
        #     if cfg.write_job_status:
        #         with open(job_status_file, "a") as f:
        #             f.write(f"Error in num_run: {num_run}.\n")
        #     continue

        if num_failed >= 10:
            print(f"Failed to complete num_run: {num_run} after 10 attempts (using seeds {seed-num_failed}-{seed}).")
            break
        

    to_pickle = {'p_lls': p_lls, 'times': times, 'num_runs': num_runs, 
                'num_samples': num_samples, 'num_tuning_samples': num_tuning_samples, 'target_accept': target_accept}


    with open(f'experiments/results/{cfg.model}/blackjax{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)

    # average over runs
    for name in moments_collection.keys():
        moments_collection[name] = np.mean(moments_collection[name], axis=1)
        
    with open(f'experiments/results/{cfg.model}/blackjax_moments{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
        pickle.dump(moments_collection, f)
        
    print()

if __name__ == "__main__":
    run_experiment()