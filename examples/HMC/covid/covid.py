import pymc as pm
import numpy as np 

nRs = 92
nWs = 21
nCMs = 11

def get_model(data, covariates):

    model = pm.Model()


    def Expected_Log_Rs(RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, log_infected, i):
        return (RegionR - CM_alpha@ActiveCMs_NPIs - Wearing_alpha*ActiveCMs_wearing - Mobility_alpha*ActiveCMs_mobility)[i]
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

        #Region level
        InitialSize_log = pm.Normal('InitialSize_log', mu=0, sigma=1, shape=nRs)
        log_infected_noise = pm.Normal('log_infe cted_noise', mu=0, sigma=1, shape=nRs)

        #Week level
        log_infecteds = [InitialSize_log]
        
        for i in range(nWs):
            
            log_infected = pm.Normal(f'log_infected_{i}', mu=log_infecteds[i] + Expected_Log_Rs(RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, log_infected, i), sigma=log_infected_noise)
            log_infecteds.append(log_infected)
                   
        infected = pm.Deterministic('infected', np.exp(np.array(log_infecteds)))
        obs = pm.Poisson('obs', mu=infected, observed=true_obs)
        
        # obs = pm.Bernoulli("obs", logit_p=logits, shape=(num_actors, num_blocks, num_repeats_plate), observed=true_obs)
            
    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['plate_nWs']:]}

    test_covariates = all_covariates

    test_covariates['ActiveCMs_NPIs'] = test_covariates['ActiveCMs_NPIs'][:,platesizes['plate_nWs']:, :]
    test_covariates['ActiveCMs_wearing'] = test_covariates['ActiveCMs_wearing'][:,platesizes['plate_nWs']:]
    test_covariates['ActiveCMs_mobility'] = test_covariates['ActiveCMs_mobility'][:,platesizes['plate_nWs']:]

    return {**test_data, **test_covariates, 'num_repeats_plate': num_repeats_extended - num_repeats}