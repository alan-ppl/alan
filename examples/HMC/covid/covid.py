import pymc as pm
import numpy as np 
import math 
#import theano.tensor as T
import pytensor.tensor as pt

nRs = 92
nDs = 137
nCMs = 11

def get_model(data, covariates):

    model = pm.Model()


    def Expected_Log_Rs(RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, i):       
        return (RegionR + ActiveCMs_NPIs@CM_alpha + Wearing_alpha*ActiveCMs_wearing + Mobility_alpha*ActiveCMs_mobility)[:,i]
    
    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])

        # covariates
        ActiveCMs_NPIs = pm.MutableData('ActiveCMs_NPIs', covariates['ActiveCMs_NPIs'])
        ActiveCMs_wearing = pm.MutableData('ActiveCMs_wearing', covariates['ActiveCMs_wearing'])
        ActiveCMs_mobility = pm.MutableData('ActiveCMs_mobility', covariates['ActiveCMs_mobility'])

        #constants
        cm_prior_scale=1
        wearing_mean=0
        wearing_sigma=0.4
        mobility_mean=1.704
        mobility_sigma=0.44
        R_prior_mean_mean=1.07
        R_prior_mean_scale=0.2
        R_noise_scale=0.4

        
        #Global
        CM_alpha = pm.Normal("CM_alpha", mu=0, sigma=cm_prior_scale, shape=nCMs-2)
        Wearing_alpha = pm.Normal("Wearing_alpha", mu=wearing_mean, sigma=wearing_sigma)
        Mobility_alpha = pm.Normal("Mobility_alpha", mu=mobility_mean, sigma=mobility_sigma)
        RegionR = pm.Normal("RegionR", mu=R_prior_mean_mean, sigma=R_prior_mean_scale + R_noise_scale)
        InitialSize_log_mean = pm.Normal('InitialSize_log_mean', mu=math.log(1000), sigma=0.5)
        log_infected_noise_mean = pm.Normal('log_infected_noise_mean', mu=0, sigma=0.25)

        #Region level
        InitialSize_log = pm.Normal('InitialSize_log', mu=InitialSize_log_mean, sigma=0.5, shape=nRs)
        log_infected_noise = pm.Normal('log_infected_noise', mu=log_infected_noise_mean, sigma=0.25, shape=nRs)
        psi = pm.Normal('psi', mu=math.log(1000), sigma=1, shape=nRs)

        #Days 
        log_infecteds = [InitialSize_log]
        for i in range(data['obs'].shape[1]):
            log_infected = pm.Normal(f'log_infected_{i}', mu=log_infecteds[i] + Expected_Log_Rs(RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, i), sigma=pm.math.exp(log_infected_noise))
            log_infecteds.append(log_infected)
        
        log_infected = pm.Deterministic("log_infected", pt.stack(log_infecteds[1:], axis=1))
        
        obs = pm.NegativeBinomial('obs', alpha=pt.exp(pt.reshape(psi, (nRs, 1))), mu=pt.exp(log_infected), observed=true_obs)
        
            
    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['nDs']:]}

    test_covariates = all_covariates

    test_covariates['ActiveCMs_NPIs'] = test_covariates['ActiveCMs_NPIs'][:,platesizes['nDs']:, :]
    test_covariates['ActiveCMs_wearing'] = test_covariates['ActiveCMs_wearing'][:,platesizes['nDs']:]
    test_covariates['ActiveCMs_mobility'] = test_covariates['ActiveCMs_mobility'][:,platesizes['nDs']:]

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