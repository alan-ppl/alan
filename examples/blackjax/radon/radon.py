import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

States=7

Counties= 5 #10 in total, half held back for testing
Zips = 5


def get_model(data, covariates):

    params = namedtuple("model_params", ["global_mean", "global_log_sigma", "State_mean", "State_log_sigma", "County_mean", "Beta_u", "Beta_basement", "County_log_sigma"])
    def joint_logdensity(params, data, covariates):
        #Global level
        global_mean = stats.norm.logpdf(params.global_mean, 0., 1.).sum()
        global_log_sigma = stats.norm.logpdf(params.global_log_sigma, 0., 1.).sum()
        #State level
        State_mean = stats.norm.logpdf(params.State_mean, params.global_mean, jnp.exp(params.global_log_sigma)).sum()
        State_log_sigma = stats.norm.logpdf(params.State_log_sigma, 0., 1.).sum()
        #County level
        County_mean = stats.norm.logpdf(params.County_mean.transpose(), params.State_mean, jnp.exp(params.State_log_sigma)).sum()
        Beta_u = stats.norm.logpdf(params.Beta_u, 0., 1.).sum()
        Beta_basement = stats.norm.logpdf(params.Beta_basement, 0., 1.).sum()
        County_log_sigma = stats.norm.logpdf(params.County_log_sigma, 0., 1.).sum()
        #Zip level
        obs = stats.norm.logpdf(data.reshape(5,7,5), 1000*params.County_mean + 10*covariates['basement'].reshape(5,7,5)*params.Beta_basement + covariates['log_uranium'].reshape(5,7,5) * params.Beta_u, jnp.exp(params.County_log_sigma)).sum()
        return global_mean + global_log_sigma + State_mean + State_log_sigma + County_mean + Beta_u + Beta_basement + County_log_sigma + obs
    
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
            County_mean=jax.random.normal(key5, shape=(States,Counties)),
            Beta_u=jax.random.normal(key6, shape=(States,Counties)),
            Beta_basement=jax.random.normal(key7, shape=(States,Counties)),
            County_log_sigma=jax.random.normal(key8, shape=(States,Counties)),
        )

    return joint_logdensity, params, init_param_fn

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['Counties']:]}

    test_covariates = all_covariates
    test_covariates['basement'] = test_covariates['basement'][:,platesizes['Counties']:]
    test_covariates['log_uranium'] = test_covariates['log_uranium'][:,platesizes['Counties']:]

    return test_data, test_covariates