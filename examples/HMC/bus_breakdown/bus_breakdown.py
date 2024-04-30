import pymc as pm
import numpy as np 

M, J, I = 3, 3, 30

def get_model(data, covariates):
    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    model = pm.Model()

    with model:
        true_obs = pm.MutableData('true_obs', data['obs'].transpose(2,1,0))

        # Year level
        sigma_beta = pm.Normal('sigma_beta', mu=0, sigma=1)
        mu_beta    = pm.Normal('mu_beta', mu=0, sigma=1)
        beta       = pm.Normal('beta', mu=mu_beta, sigma=np.exp(sigma_beta), shape=M)
        sigma_alpha = pm.Normal('sigma_alpha', mu=0, sigma=1, shape=M)
        # Borough level
        
        alpha = pm.Normal('alpha', mu=beta, sigma=np.sqrt(np.exp(sigma_alpha)), shape=(J,M))

        # ID level
        log_sigma_phi_psi = pm.Normal('log_sigma_phi_psi', mu=0, sigma=1)
        
        # psi = pm.MvNormal('psi', mu=np.zeros(run_type_dim), cov=np.exp(log_sigma_phi_psi)*np.eye(run_type_dim), shape=(run_type_dim,))
        # phi = pm.MvNormal('phi', mu=np.zeros(bus_company_name_dim), cov=np.exp(log_sigma_phi_psi)*np.eye(bus_company_name_dim), shape=(bus_company_name_dim,))

        psi = pm.Normal('psi', mu=0, sigma=log_sigma_phi_psi.exp(), shape=run_type_dim)
        phi = pm.Normal('phi', mu=0, sigma=log_sigma_phi_psi.exp(), shape=bus_company_name_dim)
        
        # Covariates
        bus_company_name = pm.MutableData('bus_company_name', covariates['bus_company_name'])
        run_type         = pm.MutableData('run_type', covariates['run_type'])
        alph = pm.Normal('alph', mu=0, sigma=np.log(10), shape=(I,J,M))
        log_delay = pm.Normal('log_delay', mu=alpha + ((bus_company_name @ phi) + (run_type @ psi)).reshape((I,J,M)), sigma=1, shape=(I, J, M))
        
        probs = pm.Deterministic('logits', pm.math.maximum(pm.math.sigmoid(log_delay), 0.001)) 
        obs = pm.NegativeBinomial('obs', alpha=np.exp(alph) ** 2, mu = probs, observed=true_obs, shape=(I, J, M))

    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,:,platesizes['plate_ID']:].transpose(2,1,0)}

    test_covariates = all_covariates
    for key in test_covariates:
        test_covariates[key] = test_covariates[key][:,:,platesizes['plate_ID']:]

    return {**test_data, **test_covariates}

if __name__ == '__main__':
    #Test model without jax
    import pickle
    
    with open(f'data/real_data.pkl', 'rb') as f:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
    
    model = get_model(data, covariates)
    with model:
        # trace = pm.sample(1, tune=1, chains=1)
        prior = pm.sample_prior_predictive(samples=2000)
        
        moments_collection = {}
        print(prior)
        # compute moments for each latent
        for name in latent_names:
            print(name)
            print(prior.prior[name].mean(("chain", "draw")))
            print(prior.prior[name].std(("chain", "draw")))
            
        for name in ['obs']:
            print(name)
            print(prior.prior_predictive[name].mean(("chain", "draw")))
            print(prior.prior_predictive[name].std(("chain", "draw")))