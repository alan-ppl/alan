import torch as t
from alan import Split, mean, mean2
import pickle
import hydra
import importlib.util
import sys
from pathlib import Path

from itertools import product

from matplotlib import pyplot as plt

@hydra.main(version_base=None, config_path='config', config_name='conf_QQ')
def run_experiment(cfg):
    print(cfg)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print("Device:", device)

    K = cfg.K
    lr = cfg.lr
    num_samples = cfg.num_samples
    num_iters = cfg.num_iters


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


    spec = importlib.util.spec_from_file_location(cfg.model, f"{cfg.model}/{model_name}.py")
    model = importlib.util.module_from_spec(spec)
    sys.modules[cfg.model] = model
    spec.loader.exec_module(model)

    # Make sure all the required folders exist for this model
    Path(f"QQ_plots/{model_name}").mkdir(parents=True, exist_ok=True)

    platesizes, all_platesizes, data, all_data, covariates, all_covariates = model.load_data_covariates(device, 0, f'{cfg.model}/data/')

    # Put extended data and covariates on device
    for key in all_data:
        all_data[key] = all_data[key].to(device)
    for key in all_covariates:
        all_covariates[key] = all_covariates[key].to(device)

    t.manual_seed(0)
    print(f"K: {K}, lr: {lr}")

    
    
    P = model.get_P(platesizes, covariates)
    latent_names = list(P.varname2groupvarname().keys())
    latent_names.remove('obs')

    moment_list = list(product(latent_names, [mean, mean2]))
    
    prior_latent_samples = P.sample(num_samples)
    

    plate_names_obs = prior_latent_samples['obs'].names
    data_samples = prior_latent_samples['obs'].rename(*plate_names_obs[:-1], None)

    prior_means = {name: prior_latent_samples[name].mean('N').rename(None) for name in latent_names}
    prior_vars = {name: prior_latent_samples[name].var('N').rename(None) for name in latent_names}
    
    post_means = {name: [] for name in latent_names}
    post_vars = {name: [] for name in latent_names}
    post_samples = {name: [] for name in latent_names}
    for i in range(num_samples):
        prob = model.generate_problem(device, platesizes, {'obs': data_samples[...,i]}, covariates, Q_param_type='qem')
        print(f"Sample {i}")
        for _ in range(num_iters):
            sample = prob.sample(K, reparam=False)
            sample.update_qem_params(0.1)
            
        posterior_samples = prob.sample(K)
        posterior_latent_samples = posterior_samples.importance_sample(1)
        N_dim = posterior_latent_samples.Ndim
        posterior_means = posterior_samples.moments(moment_list)
        temp_means = [posterior_means[i] for i in range(0,len(latent_names)*2,2)]
        temp_means2 = [posterior_means[i] for i in range(1,len(latent_names)*2,2)]

        for j, k in enumerate(latent_names):
            post_means[k].append(temp_means[j].rename(None))
            post_vars[k].append(temp_means2[j].rename(None) - temp_means[j].rename(None)**2)
            post_samples[k].append(posterior_latent_samples.samples_flatdict[k].order(N_dim))

    overall_post_means = {name: t.stack(post_means[name]).mean(0).detach().rename(None) for name in latent_names}
    overall_post_vars = {name: t.stack(post_means[name]).var(0).detach().rename(None) + t.stack(post_means[name]).mean(0).detach().rename(None) for name in latent_names}

    pickle_dict = {'prior_means': prior_means, 'prior_vars': prior_vars, 'post_means': overall_post_means, 'post_vars': overall_post_vars} 
    
    with open(f"QQ_plots/{model_name}/QQ_plot_data.pkl", 'wb') as f:
        pickle.dump(pickle_dict, f)
    



    


if __name__ == "__main__":
    run_experiment()