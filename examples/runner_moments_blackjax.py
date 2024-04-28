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

import logging
logger = logging.getLogger('pymc')
logger.setLevel(logging.ERROR)

def safe_time(device):
    # if device == 'cuda':
        # t.cuda.synchronize()
    return time.time()

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
    for folder in ['results/moments', 'job_status/moments', 'plots/moments']:
        Path(f"{cfg.model}/{folder}").mkdir(parents=True, exist_ok=True)

    # t.manual_seed(0)
    if not fake_data:
        # platesizes, all_platesizes, data, all_data, covariates, all_covariates = alan_model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', False)
        with open(f'blackjax/{cfg.model}/data/real_data.pkl', 'rb') as f:
            platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
    else:
        with open(f'blackjax/{cfg.model}/data/fake_data.pkl', 'rb') as f:
            platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, latent_names = pickle.load(f)
        # platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, _ = alan_model.load_data_covariates(device, dataset_seed, f'{cfg.model}/data/', True, return_fake_latents=True)



    times = {"moments": np.zeros((num_samples, num_runs)),
             "p_ll":   np.zeros((num_samples, num_runs))}

    job_status_file = f"{cfg.model}/job_status/moments/blackjax{'_FAKE_DATA' if fake_data else ''}_status.txt"
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

            joint_logdensity, params, init_param_fn = model.get_model(data, covariates)
            rng_key = jax.random.PRNGKey(num_run)
            rng_key, init_key = jax.random.split(rng_key)
            
            warmup = blackjax.window_adaptation(blackjax.nuts, joint_logdensity)
            # we use 4 chains for sampling
            n_chains = cfg.n_chains
            rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
            init_keys = jax.random.split(init_key, n_chains)
            init_params = jax.vmap(init_param_fn)(init_keys)

            @jax.vmap
            def call_warmup(seed, param):
                (initial_states, tuned_params), _ = warmup.run(seed, param, 1000)
                return initial_states, tuned_params

            
            warmup_keys = jax.random.split(warmup_key, n_chains)
            if n_chains == 1:
                warmup_keys = warmup_keys[0]

            initial_states, tuned_params = jax.jit(call_warmup)(warmup_keys, init_params)

            def inference_loop_multiple_chains(
                rng_key, initial_states, tuned_params, log_prob_fn, num_samples, num_chains
            ):
                kernel = blackjax.nuts.build_kernel()

                def step_fn(key, state, **params):
                    return kernel(key, state, log_prob_fn, **params)

                def one_step(states, rng_key):
                    keys = jax.random.split(rng_key, num_chains)
                    states, infos = jax.vmap(step_fn)(keys, states, **tuned_params)
                    return states, (states, infos)

                keys = jax.random.split(rng_key, num_samples)
                _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

                return (states, infos)

            rng_key = jax.random.PRNGKey(num_run)
            rng_key, sample_key = jax.random.split(rng_key)
            states, infos = inference_loop_multiple_chains(
                sample_key, initial_states, tuned_params, joint_logdensity, num_samples, n_chains
            )

            states = states.position._asdict()
            #HMC means
            HMC_means = {key: np.mean(states[key], axis=0) for key in states}

            # compute moments for each latent
            for name in latent_names:
                if num_run == 0:
                    latent_shape = HMC_means[name].shape
                    moments_collection[name] = np.zeros((num_samples, num_runs, *latent_shape))
                
                moments_collection[name][:, num_run, ...] = np.array([states[name][j,...] for j in range(1, num_samples+1)])
                    
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

    for i, name in enumerate(latent_names):
        if fake_data:
            ground_truth = fake_latents[name]
            latent_ndim = ground_truth.ndim # no need for -1 since we haven't yet added the iteration dimension

            # if (None, None, *ground_truth.names) != moments_collection[name].names:
            #     ground_truth = ground_truth.align_as(moments_collection[name]).mean(1)
            #     latent_ndim = ground_truth.ndim - 1
        else:
            ground_truth = moments_collection[name].mean(1)
            latent_ndim = ground_truth.ndim - 1
        
        


    to_pickle = {'times': times, 'num_runs': num_runs, 
                'num_samples': num_samples, 'num_tuning_samples': num_tuning_samples, 'target_accept': target_accept}


    with open(f'{cfg.model}/results/moments/blackjax{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)

    with open(f'{cfg.model}/results/moments/blackjax_moments{dataset_seed}{"_FAKE_DATA" if fake_data else ""}.pkl', 'wb') as f:
        pickle.dump(moments_collection, f)
        
    print()

if __name__ == "__main__":
    run_experiment()