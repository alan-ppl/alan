import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

M, J, I, Returns = 6, 12, 200, 5
I_extended = 300

def get_model(data, covariates):

    params = namedtuple("model_params", ["bird_mean_mean", "bird_mean_log_var", "alpha_mean", "alpha_log_var", "beta_mean", "beta_log_var", "bird_mean", "alpha", "beta", "bird_year_mean", "z"])
    def joint_logdensity(params, data, covariates):
        bird_mean_mean = stats.norm.logpdf(params.bird_mean_mean, 0., 1.).sum()
        bird_mean_log_var = stats.norm.logpdf(params.bird_mean_log_var, 0., 1.).sum()
        
        alpha_mean = stats.norm.logpdf(params.alpha_mean, 0., 1.).sum()
        alpha_log_var = stats.norm.logpdf(params.alpha_log_var, 0., 1.).sum()
        
        beta_mean = stats.norm.logpdf(params.beta_mean, 0., 1.).sum()
        beta_log_var = stats.norm.logpdf(params.beta_log_var, 0., 1.).sum()
        
        bird_mean = stats.norm.logpdf(params.bird_mean, params.bird_mean_mean, jnp.exp(params.bird_mean_log_var)).sum()
        alpha = stats.norm.logpdf(params.alpha, params.alpha_mean, jnp.exp(params.alpha_log_var)).sum()
        beta = stats.norm.logpdf(params.beta, params.beta_mean, jnp.exp(params.beta_log_var)).sum()
        
        bird_year_mean = stats.norm.logpdf(params.bird_year_mean, params.bird_mean, 1.).sum()
        
        z = stats.bernoulli.logpmf(params.z, jax.nn.softmax(params.bird_mean * params.beta * covariates['weather'].transpose(2,0,1))).sum()
        
        obs = stats.bernoulli.logpmf(data.transpose(3,2,0,1), jax.nn.softmax(params.alpha * covariates['quality'].transpose(2,0,1) * params.z + (1-params.z)*(-10))).sum()
        
        return bird_mean_mean + bird_mean_log_var + alpha_mean + alpha_log_var + beta_mean + beta_log_var + bird_mean + alpha + beta + bird_year_mean + z + obs
    
    def joint_logdensity_pred_ll(params, data, covariates):
        bird_mean_mean = stats.norm.logpdf(params.bird_mean_mean, 0., 1.).sum()
        bird_mean_log_var = stats.norm.logpdf(params.bird_mean_log_var, 0., 1.).sum()
        
        alpha_mean = stats.norm.logpdf(params.alpha_mean, 0., 1.).sum()
        alpha_log_var = stats.norm.logpdf(params.alpha_log_var, 0., 1.).sum()
        
        beta_mean = stats.norm.logpdf(params.beta_mean, 0., 1.).sum()
        beta_log_var = stats.norm.logpdf(params.beta_log_var, 0., 1.).sum()
        
        bird_mean = stats.norm.logpdf(params.bird_mean, params.bird_mean_mean, jnp.exp(params.bird_mean_log_var)).sum()
        alpha = stats.norm.logpdf(params.alpha, params.alpha_mean, jnp.exp(params.alpha_log_var)).sum()
        beta = stats.norm.logpdf(params.beta, params.beta_mean, jnp.exp(params.beta_log_var)).sum()
        
        bird_year_mean = stats.norm.logpdf(params.bird_year_mean, params.bird_mean, 1.).sum()
        
        z = stats.bernoulli.logpmf(params.z, jax.nn.softmax(params.bird_mean * params.beta * covariates['weather'].transpose(2,0,1))).sum()
        
        obs = stats.bernoulli.logpmf(data.transpose(3,2,0,1), jax.nn.softmax(params.alpha * covariates['quality'].transpose(2,0,1) * params.z + (1-params.z)*(-10))).sum()
    
        return obs
    
    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11 = jax.random.split(seed, 11)
        return params(
            bird_mean_mean = jax.random.normal(key1),
            bird_mean_log_var = jax.random.normal(key2),
            alpha_mean = jax.random.normal(key3),
            alpha_log_var = jax.random.normal(key4),
            beta_mean = jax.random.normal(key5),
            beta_log_var = jax.random.normal(key6),
            bird_mean = jax.random.normal(key7, (J,)),
            alpha = jax.random.normal(key8, (J,)),
            beta = jax.random.normal(key9, (J,)),
            bird_year_mean = jax.random.normal(key10, (M,J)),
            z = jax.random.normal(key11, (I,M,J))
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


