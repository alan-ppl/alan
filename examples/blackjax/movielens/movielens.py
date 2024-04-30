import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

d_z = 18
M, N = 300, 5

def get_model(data, covariates):

    params = namedtuple("model_params", ["mu_z", "psi_z", "z"])
    def joint_logdensity(params):
        mu_z = stats.norm.logpdf(params.mu_z, 0., 1.).sum()
        psi_z = stats.norm.logpdf(params.psi_z, 0., 1.).sum()
        z = stats.norm.logpdf(params.z, params.mu_z, jnp.exp(params.psi_z)).sum()
        obs = stats.bernoulli.logpmf(data['obs'], jax.nn.sigmoid((params.z @ covariates['x']))).sum()
        
        return mu_z + psi_z + z + obs
    
    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key1, key2, key3 = jax.random.split(seed, 3)
        return params(
            mu_z=jax.random.normal(key1),
            psi_z=jax.random.normal(key2),
            z=jax.random.normal(key3, shape=(M,d_z)),
        )
    
    return joint_logdensity, params, init_param_fn

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['plate_2']:]}

    test_covariates = all_covariates
    test_covariates['x'] = test_covariates['x'][:,platesizes['plate_2']:]

    return {**test_data, **test_covariates}

if __name__ == '__main__':
    #Test model without jax
    import pickle
    
    with open(f'data/real_data.pkl', 'rb') as f:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
    
    model = get_model(data, covariates)
    with model:
        # trace = pm.sample(1, tune=1, chains=1)
        prior = pm.sample_prior_predictive(samples=2000)
        
        moments_collection = {}
        print(prior)
        # compute moments for each latent
        for name in latent_names:
            print(name)
            print(prior.prior[name].mean(("chain", "draw")))
            print(prior.prior[name].std(("chain", "draw")))
            
        for name in ['obs']:
            print(name)
            print(prior.prior_predictive[name].mean(("chain", "draw")))
            print(prior.prior_predictive[name].std(("chain", "draw")))