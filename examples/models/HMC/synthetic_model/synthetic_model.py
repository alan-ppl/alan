import pymc as pm
import numpy as np 

N = 4
z_mean = 33
z_var = 0.5
obs_var = 10
def get_model(data, covariates):
    model = pm.Model()
    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])
        
        mean = pm.Normal('mean', mu=z_mean, sigma=z_var)
        
        obs = pm.Normal('obs',  mu=mean, sigma=obs_var, observed=true_obs, shape=(N,))

    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][platesizes['plate_1']:]}

    test_covariates = all_covariates

    return {**test_data, **test_covariates}

if __name__ == '__main__':
    #Test model without jax
    import pickle
    
    with open(f'data/real_data.pkl', 'rb') as f:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
    
    model = get_model(data, covariates)
    with model:
        trace = pm.sample(10, tune=10, chains=1)
        print(trace)