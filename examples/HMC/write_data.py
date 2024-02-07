import torch as t
import importlib
import sys
from pathlib import Path
import pickle

def write_data(model, fake_data=True, dataset_seed=0):
    device = 'cpu'
    
    spec = importlib.util.spec_from_file_location(model, f"../{model}/{model}.py")
    alan_model = importlib.util.module_from_spec(spec)
    sys.modules[model] = alan_model
    spec.loader.exec_module(alan_model)

    Path(f"{model}/data").mkdir(parents=True, exist_ok=True)

    t.manual_seed(0)
    if not fake_data:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates = alan_model.load_data_covariates(device, dataset_seed, f'../{model}/data/', False)
    else:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, _ = alan_model.load_data_covariates(device, dataset_seed, f'../{model}/data/', True, return_fake_latents=True)

    temp_P = alan_model.get_P(platesizes, covariates)
    latent_names = list(temp_P.varname2groupvarname().keys())
    latent_names.remove('obs')

    # convert from torch to numpy
    for data_covs in [data, all_data, covariates, all_covariates]:
        for key in data_covs:
            data_covs[key] = data_covs[key].numpy()

    if fake_data:
        for key in fake_latents:
            fake_latents[key] = fake_latents[key].numpy()

    if not fake_data:
        with open(f'{model}/data/real_data.pkl', 'wb') as f:
            pickle.dump((platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names), f)
    else:
        with open(f'{model}/data/fake_data.pkl', 'wb') as f:
            pickle.dump((platesizes, all_platesizes, data, all_data, covariates, all_covariates, fake_latents, latent_names), f)

if __name__ == '__main__':
    for model in ['bus_breakdown', 'movielens', 'chimpanzees']:
        for fake_data in [True, False]:
            write_data(model, fake_data=fake_data)