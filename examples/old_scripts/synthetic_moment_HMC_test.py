import pymc as pm
import numpy as np 

N = 20
M= 20
z_mean = 33
z_var = 0.5
obs_var = 10
def get_model(data, covariates):
    model = pm.Model()
    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])
        
        mean = pm.Normal('mean', mu=z_mean, sigma=z_var)
        var = pm.Normal('var', mu=0., sigma=1.)
        
        obs_mean = pm.Normal('obs_mean', mu=mean, sigma=var.exp(), shape=(M,))
        obs_var = pm.Normal('obs_var', mu=mean, sigma=var.exp(), shape=(M,))
        
        obs = pm.Normal('obs', mu = obs_mean, sigma=obs_var.exp(), observed=true_obs, shape=(M,N))

    return model

def get_test_data_cov_dict(all_data, all_covariates, platesizes):
    test_data = all_data
    test_data = {'true_obs': all_data['obs'][platesizes['plate_obs']:]}

    test_covariates = all_covariates

    return {**test_data, **test_covariates}

if __name__ == '__main__':
    #Test model without jax
    import pickle
    
    with open(f'fake_data.pkl', 'rb') as f:
        platesizes, all_platesizes, data, all_data, covariates, all_covariates, latent_names = pickle.load(f)
    
    latent_names = ['mean', 'var', 'obs_mean', 'obs_var']
    model = get_model(data, covariates)
    with model:
        trace = pm.sample(20000, tune=100, chains=10)
        
        moments_collection = {}
        # compute moments for each latent
        for name in latent_names:
            latent_shape = trace.posterior[name].mean(("chain", "draw")).shape
            moments_collection[name] = np.zeros((100, *latent_shape))
            
            moments_collection[name][:, ...] = np.array([trace.posterior[name][:,:j].mean(("chain", "draw")).data for j in range(1, 100+1)])

for name in latent_names:
    print(name)
    print(moments_collection[name].mean(0))
