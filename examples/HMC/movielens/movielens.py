import pymc as pm
import numpy as np 

d_z = 18
M, N = 300, 5

def get_model(data, covariates):
    model = pm.Model()
    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])
        
        # mu_z = pm.MvNormal('mu_z', mu=np.zeros(d_z), cov=np.eye(d_z))
        mu_z = pm.Normal('mu_z', mu=0, sigma=1, shape=d_z)

        # psi_z = pm.MvNormal('psi_z', mu=np.zeros(d_z), cov=np.eye(d_z))
        psi_z = pm.Normal('psi_z', mu=0, sigma=1, shape=d_z)

        # z = pm.MvNormal('z', mu=mu_z, cov=np.eye(d_z)*psi_z.exp(), shape=(M, d_z))
        z = pm.Normal('z', mu=mu_z, sigma=psi_z.exp(), shape=(M, d_z))

        x = pm.MutableData('x', covariates['x'])

        logits = pm.Deterministic('logits', (z @ x.transpose(0,2,1)).diagonal().transpose())
        # ^^ equivalent to:
        # logits = pm.Deterministic('logits', np.einsum('ij,ikj->ik', z, x))
        # but pymc doesn't like einsums in Deterministic nodes
        
        obs = pm.Bernoulli('obs', logit_p = logits, observed=true_obs, shape=(M, N))

    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][:,platesizes['plate_2']:]}

    test_covariates = all_covariates
    test_covariates['x'] = test_covariates['x'][:,platesizes['plate_2']:]

    return {**test_data, **test_covariates}