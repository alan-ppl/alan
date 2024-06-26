import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

States=4

Zips = 150


def get_model(data, covariates):

    params = namedtuple("model_params", ["global_mean", "global_log_sigma", "State_mean", "State_log_sigma", "Beta_u", "Beta_basement"])
    def joint_logdensity(params, data, covariates):
        #Global level
        global_mean = stats.norm.logpdf(params.global_mean, 0., 1.).sum()
        global_log_sigma = stats.norm.logpdf(params.global_log_sigma, 0., 1.).sum()
        #State level
        State_mean = stats.norm.logpdf(params.State_mean, params.global_mean,jnp.exp(params.global_log_sigma)).sum()
        #State_mean = params.global_mean + params.State_mean_non_cent * jnp.exp(params.global_log_sigma)
        State_log_sigma = stats.norm.logpdf(params.State_log_sigma, 0., 1.).sum()

        Beta_u = stats.norm.logpdf(params.Beta_u, 0., 1.).sum()
        Beta_basement = stats.norm.logpdf(params.Beta_basement, 0., 1.).sum()
        #Zip level

        
        obs = stats.norm.logpdf(data, params.State_mean[:,jnp.newaxis] + covariates['basement']*params.Beta_basement[:,jnp.newaxis] + covariates['log_uranium'] * params.Beta_u[:,jnp.newaxis], jnp.exp(params.State_log_sigma[:,jnp.newaxis])).sum()

        return global_mean + global_log_sigma + State_mean + State_log_sigma  + Beta_u + Beta_basement  + obs
    
    
    def joint_logdensity_pred_ll(params, test_data, test_covariates):
        #Global level
        global_mean = stats.norm.logpdf(params.global_mean, 0., 1.).sum()
        global_log_sigma = stats.norm.logpdf(params.global_log_sigma, 0., 1.).sum()
        #State level
        State_mean = stats.norm.logpdf(params.State_mean, params.global_mean,jnp.exp(params.global_log_sigma)).sum()
        #State_mean = params.global_mean + params.State_mean_non_cent * jnp.exp(params.global_log_sigma)
        State_log_sigma = stats.norm.logpdf(params.State_log_sigma, 0., 1.).sum()
        Beta_u = stats.norm.logpdf(params.Beta_u, 0., 1.).sum()
        Beta_basement = stats.norm.logpdf(params.Beta_basement, 0., 1.).sum()
        #Zip level
        obs = stats.norm.logpdf(test_data, params.State_mean[:,jnp.newaxis] + test_covariates['basement']*params.Beta_basement[:,jnp.newaxis] + test_covariates['log_uranium'] * params.Beta_u[:,jnp.newaxis], jnp.exp(params.State_log_sigma[:,jnp.newaxis])).sum()
        return obs
    
    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key1, key2, key3, key4, key5, key6, key7, key8 = jax.random.split(seed, 8)
        return params(
            global_mean=jax.random.normal(key1),
            global_log_sigma=jax.random.normal(key2),
            State_mean=jax.random.normal(key3, shape=(States,)),
            State_log_sigma=jax.random.normal(key4, shape=(States,)),
            Beta_u=jax.random.normal(key6, shape=(States,)),
            Beta_basement=jax.random.normal(key7, shape=(States,)),
        )
        
    def transform_non_cent_to_cent(params, covariates=None):
        # params['State_mean'] = params['global_mean'][:,jnp.newaxis] + params['State_mean_non_cent'] * jnp.exp(params['global_log_sigma'][:,jnp.newaxis])
        # params['County_mean'] = params['State_mean'][...,jnp.newaxis] + params['County_mean_non_cent'] * jnp.exp(params['State_log_sigma'][...,jnp.newaxis])
        
        # del params['State_mean_non_cent']
        # del params['County_mean_non_cent']
        return params

    return joint_logdensity, joint_logdensity_pred_ll, params, init_param_fn, transform_non_cent_to_cent

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['Zips']:]}

    test_covariates = all_covariates
    test_covariates['basement'] = test_covariates['basement'][:,platesizes['Zips']:]
    test_covariates['log_uranium'] = test_covariates['log_uranium'][:,platesizes['Zips']:]

    return test_data, test_covariates