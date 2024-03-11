import pymc as pm
import numpy as np 

Zips = 5
Counties= 5 #10 in total, half held back for testing
States=7


def get_model(data, covariates):
    model = pm.Model()
    with model:
        #Data
        true_obs = pm.MutableData('true_obs', data['obs'])
        #Covariates
        basement = pm.MutableData('basement', covariates['basement'])
        log_uranium         = pm.MutableData('log_uranium ', covariates['log_uranium '])
        
        #Model
        #Global level
        global_mean = pm.Normal('global_mean', mu=0, sigma=1)
        global_log_sigma = pm.Normal('global_log_sigma', mu=0, sigma=1)
        #State level
        State_mean = pm.Normal('State_mean', mu=global_mean, sigma=global_log_sigma.exp(), shape=(States,))
        State_log_sigma = pm.Normal('State_log_sigma', mu=0, sigma=1, shape=(States,))
        #County level
        County_mean = pm.Normal('County_mean', mu=State_mean, sigma=State_log_sigma.exp(), shape=(States,Counties))
        County_log_sigma = pm.Normal('County_log_sigma', mu=0, sigma=1, shape=(States,Counties))
        Beta_u = pm.Normal('Beta_u', mu=0, sigma=1, shape=(States,Counties))
        Beta_basement = pm.Normal('Beta_basement', mu=0, sigma=10, shape=(States,Counties))
        #Zip level
        obs = pm.Normal('obs', mu=1000*County_mean + 10*basement*Beta_basement + log_uranium * Beta_u, sigma=County_log_sigma.exp(), observed=true_obs, shape=(States,Counties,Zips))
        
    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['plate_counties']:]}

    test_covariates = all_covariates
    test_covariates['basement'] = test_covariates['basement'][:,platesizes['plate_counties']:]
    test_covariates['log_uranium '] = test_covariates['log_uranium '][:,platesizes['plate_counties']:]

    return {**test_data, **test_covariates}