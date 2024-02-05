import pymc as pm
import numpy as np 

M, J, I = 3, 3, 30

def get_model(data, covariates):
    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    model = pm.Model()

    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])

        # Year level
        sigma_beta = pm.Normal('sigma_beta', mu=0, sigma=1)
        mu_beta    = pm.Normal('mu_beta', mu=0, sigma=1)
        beta       = pm.Normal('beta', mu=mu_beta, sigma=np.exp(sigma_beta), shape=(M,))

        # Borough level
        sigma_alpha = pm.Normal('sigma_alpha', mu=0, sigma=1, shape=(J,))
        alpha       = pm.Normal('alpha', mu=beta, sigma=np.sqrt(np.exp(sigma_alpha)), shape=(M,J))

        # ID level
        log_sigma_phi_psi = pm.Normal('log_sigma_phi_psi', mu=0, sigma=1)
        
        # psi = pm.MvNormal('psi', mu=np.zeros(run_type_dim), cov=np.exp(log_sigma_phi_psi)*np.eye(run_type_dim), shape=(run_type_dim,))
        # phi = pm.MvNormal('phi', mu=np.zeros(bus_company_name_dim), cov=np.exp(log_sigma_phi_psi)*np.eye(bus_company_name_dim), shape=(bus_company_name_dim,))

        psi = pm.Normal('psi', mu=0, sigma=log_sigma_phi_psi.exp(), shape=(run_type_dim,))
        phi = pm.Normal('phi', mu=0, sigma=log_sigma_phi_psi.exp(), shape=(bus_company_name_dim,))
        
        # Covariates
        bus_company_name = pm.MutableData('bus_company_name', covariates['bus_company_name'])
        run_type         = pm.MutableData('run_type', covariates['run_type'])

        logits = pm.Deterministic('logits', (alpha + (phi @ bus_company_name.transpose(0,1,3,2) + psi @ run_type.transpose(0,1,3,2)).transpose(2,0,1)).transpose(1,2,0))

        obs = pm.NegativeBinomial('obs', n=131, p=1/(1+np.exp(-logits)), observed=true_obs, shape=(M, J, I))

    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,:,platesizes['plate_ID']:]}

    test_covariates = all_covariates
    for key in test_covariates:
        test_covariates[key] = test_covariates[key][:,:,platesizes['plate_ID']:]

    return {**test_data, **test_covariates}