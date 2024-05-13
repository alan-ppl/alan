import jax
import jax.numpy as jnp
import jax.scipy.stats as stats


import numpy as np 

from collections import namedtuple
M, J, I = 3, 3, 30

def get_model(data, covariates):
    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    params = namedtuple("model_params", ["psi", "phi", "sigma_beta", "mu_beta", "beta_non_cent", "sigma_alpha", "alpha_non_cent", "alph", "log_delay_non_cent"])
    def joint_logdensity(params, data, covariates):
        #prior
        
        psi = stats.norm.logpdf(params.psi, 0., 1.).sum()
        phi = stats.norm.logpdf(params.phi, 0., 1.).sum()
        sigma_beta = stats.norm.logpdf(params.sigma_beta, 0., 1.).sum()
        mu_beta = stats.norm.logpdf(params.mu_beta, 0., 1.).sum()
        # year level

        beta_non_cent = stats.norm.logpdf(params.beta_non_cent, 0., 1.).sum()
        beta = params.mu_beta + params.beta_non_cent * jnp.exp(params.sigma_beta)
        sigma_alpha = stats.norm.logpdf(params.sigma_alpha, 0., 1.).sum()
        
        # borough level
        alpha_non_cent = stats.norm.logpdf(params.alpha_non_cent, 0., 1.).sum()
        alpha = beta + params.alpha_non_cent * jnp.exp(params.sigma_alpha)
        
        # ID level
        alph = stats.norm.logpdf(params.sigma_alpha, 0., 1.).sum()

        log_delay_non_cent = stats.norm.logpdf(params.log_delay_non_cent, 0., 1.).sum()
        log_delay = alpha.reshape(3,3,1) + ((covariates['bus_company_name'] @ params.phi.transpose()) + (covariates['run_type'] @ params.psi.transpose())) + params.log_delay_non_cent 
        obs = stats.nbinom.logpmf(data, jnp.exp(params.alph), jax.nn.softmax(log_delay)).sum()
        
        return psi + phi + sigma_beta + mu_beta + beta_non_cent + sigma_alpha + alpha_non_cent + alph + log_delay_non_cent + obs


    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key1, key2, key3, key4, key5, key6, key7, key8, key9 = jax.random.split(seed, 9)
        return params(
            psi = jax.random.normal(key1, shape=(run_type_dim,)),
            phi = jax.random.normal(key2, shape=(bus_company_name_dim,)),
            sigma_beta=jax.random.normal(key3),
            mu_beta=jax.random.normal(key4),
            beta_non_cent=jax.random.normal(key5, shape=(M,)),
            sigma_alpha=jax.random.normal(key6, shape=(M,)),
            alpha_non_cent=jax.random.normal(key7, shape=(M,J)),
            alph=jax.random.exponential(key8, shape=(M,J,I)),
            log_delay_non_cent=jax.random.normal(key9, shape=(M,J,I)),
        )
    
    def transform_non_cent_to_cent(params, covariates):
        num_samples = params['log_delay_non_cent'].shape[0]
        params['beta'] = params['mu_beta'][:,jnp.newaxis] + params['beta_non_cent'] * jnp.exp(params['sigma_beta'][:,jnp.newaxis])
        params['alpha'] = params['beta'][:,jnp.newaxis] + params['alpha_non_cent'] * jnp.exp(params['sigma_alpha'][:,jnp.newaxis])

        params['log_delay'] = params['alpha'].reshape(num_samples,3,3,1) + ((covariates['bus_company_name'] @ params['phi'].transpose()) + (covariates['run_type'] @ params['psi'].transpose())).reshape(num_samples,3,3,30) + params['log_delay_non_cent']
        

        del params['beta_non_cent']
        del params['alpha_non_cent']
        del params['log_delay_non_cent']
        return params
        
    return joint_logdensity, params, init_param_fn, transform_non_cent_to_cent

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,:,platesizes['plate_ID']:]}

    test_covariates = all_covariates
    for key in test_covariates:
        test_covariates[key] = test_covariates[key][:,:,platesizes['plate_ID']:]

    return test_data, test_covariates