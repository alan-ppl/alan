import pymc as pm
import numpy as np 

num_actors, num_blocks = 7, 6
num_repeats, num_repeats_extended = 10, 12

def get_model(data, covariates):

    model = pm.Model()

    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])

        # covariates
        condition = pm.MutableData('condition', covariates['condition'])
        prosoc_left = pm.MutableData('prosoc_left', covariates['prosoc_left'])

        num_repeats_plate = pm.MutableData('num_repeats_plate', num_repeats)

        pre_sigma_block = pm.Cauchy("pre_sigma_block", alpha=0, beta=1)
        pre_sigma_actor = pm.Cauchy("pre_sigma_actor", alpha=0, beta=1)

        sigma_block = pm.Deterministic('sigma_block', pm.math.abs(pre_sigma_block))
        sigma_actor = pm.Deterministic('sigma_actor', pm.math.abs(pre_sigma_actor))

        # sigma_block = pm.HalfCauchy("sigma_block", beta=1),
        # sigma_actor = pm.HalfCauchy("sigma_actor", beta=1),

        beta_PC = pm.Normal("beta_PC", mu=0, sigma=10),
        beta_P = pm.Normal("beta_P", mu=0, sigma=10),

        alpha = pm.Normal("alpha", mu=0, sigma=10),

        alpha_actor = pm.Normal("alpha_actor", mu=0, sigma=sigma_actor, shape=(num_actors,)),

        alpha_block = pm.Normal("alpha_block", mu=0, sigma=sigma_block, shape=(num_actors, num_blocks)),

        beta_combined = pm.Deterministic('beta_combined', prosoc_left*(beta_P + condition*beta_PC))

        add_alpha = pm.Deterministic('add_alpha3', beta_combined.transpose(1,2,0) + alpha_actor)
        add_alpha2 = pm.Deterministic('add_alpha2', add_alpha.transpose(1,2,0) + alpha_block)
        add_alpha3 = pm.Deterministic('add_alpha', add_alpha2.transpose(1,2,0) + alpha)


        logits = pm.Deterministic('logits', add_alpha3)

        # logits = pm.Deterministic('logits', (((beta_P + beta_PC*condition)*prosoc_left + alpha).transpose(2,0,1) + alpha_block).transpose(1,2,0) + alpha_actor).transpose(0,1,2)

        # obs = pm.Bernoulli("obs", logit_p=alpha + alpha_actor + alpha_block + (beta_P + beta_PC*condition)*prosoc_left, observed=true_obs, shape=(num_actors, num_blocks, num_repeats)),
        
        # breakpoint()
        
        obs = pm.Bernoulli("obs", logit_p=logits, shape=(num_actors, num_blocks, num_repeats_plate), observed=true_obs)
            
    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,:,platesizes['plate_repeats']:]}

    test_covariates = all_covariates
    for key in test_covariates:
        test_covariates[key] = test_covariates[key][:,:,platesizes['plate_repeats']:]

    return {**test_data, **test_covariates, 'num_repeats_plate': num_repeats_extended - num_repeats}