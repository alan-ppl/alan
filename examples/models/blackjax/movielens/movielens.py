import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

d_z = 18
M, N = 300, 5

def get_model(data, covariates):

    params = namedtuple("model_params", ["mu_z", "psi_z", "z"])
    def joint_logdensity(params, data, covariates):
        mu_z = stats.norm.logpdf(params.mu_z, 0., 1.).sum()
        psi_z = stats.norm.logpdf(params.psi_z, 0., 1).sum()
        z_non_cent = stats.norm.logpdf(params.z, params.mu_z, jnp.exp(params.psi_z)).sum()
        
        logits = jnp.einsum('ij,ikj->ik', params.z, covariates['x'])
        # print(logits.shape)
        # print(logits)
        obs = stats.bernoulli.logpmf(data, jax.nn.sigmoid(logits)).sum()

        return mu_z + psi_z + z_non_cent + obs
    
    def joint_logdensity_pred_ll(params, test_data, covariates):
        mu_z = stats.norm.logpdf(params.mu_z, 0., 1.).sum()
        psi_z = stats.norm.logpdf(params.psi_z, 0., 1).sum()
        z_non_cent = stats.norm.logpdf(params.z, params.mu_z, jnp.exp(params.psi_z)).sum()
        
        logits = jnp.einsum('ij,ikj->ik', params.z, covariates['x'])
        obs = stats.bernoulli.logpmf(test_data, jax.nn.sigmoid(logits)).sum()
        
        return obs
    
    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key1, key2, key3 = jax.random.split(seed, 3)
        return params(
            mu_z=jax.random.normal(key1, shape=(d_z,)),
            psi_z=jax.random.normal(key2, shape=(d_z,)),
            z=jax.random.normal(key3, shape=(M,d_z)),
        )
    
    def transform_non_cent_to_cent(params, covariates=None):
        # num_samples = params['z_non_cent'].shape[0]
        # params['z'] = params['mu_z'].reshape(num_samples,1,18) + params['z_non_cent'] * jnp.exp(params['psi_z']).reshape(num_samples,1,18)

        # del params['z_non_cent']
        return params
    
    return joint_logdensity, joint_logdensity_pred_ll, params, init_param_fn, transform_non_cent_to_cent

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['plate_2']:]}

    test_covariates = all_covariates
    test_covariates['x'] = test_covariates['x'][:,platesizes['plate_2']:]

    return test_data, test_covariates


