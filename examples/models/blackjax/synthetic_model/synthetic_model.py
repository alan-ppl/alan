import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

N = 4
z_mean = 33.0
z_var = 0.5
obs_var = 10.0
def get_model(data, covariates):
    # model = pm.Model()
    # with model:
    #     true_obs = pm.MutableData('true_obs', data['obs'])
        
    #     mean = pm.Normal('mean', mu=z_mean, sigma=z_var)
        
    #     obs = pm.Normal('obs', mean = mean, sigma=obs_var, observed=true_obs, shape=(N,))

    # return model

    params = namedtuple("model_params", ["mean"])
    def joint_logdensity(params, data, covariates):
        mean = stats.norm.logpdf(params.mean, z_mean, z_var)
        obs = stats.norm.logpdf(data, params.mean, obs_var).sum()
        
        return mean + obs
    
    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key = jax.random.split(seed, 1)[0]
        return params(
            mean=jax.random.normal(key),
        )
    
    def transform_non_cent_to_cent(params, covariates=None):
        # params['z'] = params['mu_z'][:,jnp.newaxis, jnp.newaxis] + params['z_non_cent'] * jnp.exp(params['psi_z'][:,jnp.newaxis, jnp.newaxis])
        
        # del params['z_non_cent']
        return params
    
    return joint_logdensity, params, init_param_fn, transform_non_cent_to_cent



def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][platesizes['plate_1']:]}

    test_covariates = all_covariates

    # return {**test_data, **test_covariates}
    return test_data, test_covariates

# if __name__ == '__main__':
#     #Test model without jax
#     import pickle
    
#     with open(f'data/real_data.pkl', 'rb') as f:
#         platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
    
#     model = get_model(data, covariates)
#     with model:
#         trace = pm.sample(10, tune=10, chains=1)
#         print(trace)