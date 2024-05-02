import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np 

from collections import namedtuple

nRs = 92
nDs = 137
nCMs = 11

#constants
cm_prior_scale=1
wearing_mean=0
wearing_sigma=0.4
mobility_mean=1.704
mobility_sigma=0.44
R_prior_mean_mean=1.07
R_prior_mean_scale=0.2
R_noise_scale=0.4

def get_model(data, covariates):

    model = pm.Model()


    # def Expected_Log_Rs(RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, i):       
    #     return (RegionR + ActiveCMs_NPIs@CM_alpha + Wearing_alpha*ActiveCMs_wearing + Mobility_alpha*ActiveCMs_mobility)[:,i]
    
    with model:
        true_obs = pm.MutableData('true_obs', data['obs'])

        # covariates
        ActiveCMs_NPIs = pm.MutableData('ActiveCMs_NPIs', covariates['ActiveCMs_NPIs'])
        ActiveCMs_wearing = pm.MutableData('ActiveCMs_wearing', covariates['ActiveCMs_wearing'])
        ActiveCMs_mobility = pm.MutableData('ActiveCMs_mobility', covariates['ActiveCMs_mobility'])

        #constants
       

        
        #Global
        CM_alpha = pm.Normal("CM_alpha", mu=0, sigma=cm_prior_scale, shape=nCMs-2)
        Wearing_alpha = pm.Normal("Wearing_alpha", mu=wearing_mean, sigma=wearing_sigma)
        Mobility_alpha = pm.Normal("Mobility_alpha", mu=mobility_mean, sigma=mobility_sigma)
        RegionR = pm.Normal("RegionR", mu=R_prior_mean_mean, sigma=R_prior_mean_scale + R_noise_scale)
        InitialSize_log_mean = pm.Normal('InitialSize_log_mean', mu=math.log(1000), sigma=0.5)
        log_infected_noise_mean = pm.Normal('log_infected_noise_mean', mu=0, sigma=0.25)

        #Region level
        InitialSize_log = pm.Normal('InitialSize_log', mu=InitialSize_log_mean, sigma=0.5, shape=nRs)
        # log_infected_noise = pm.Normal('log_infected_noise', mu=log_infected_noise_mean, sigma=0.25, shape=nRs)
        psi = pm.Normal('psi', mu=math.log(1000), sigma=1, shape=nRs)

        #Days 
        log_infected_noise = pm.Normal("log_infected_noise", mu = 0, sigma=0.25 / 10, shape=(nRs, data['obs'].shape[1]))
        # log_infecteds = [InitialSize_log]
        # for i in range(data['obs'].shape[1]):
        #     log_infected = pm.Normal(f'log_infected_{i}', mu=log_infecteds[i] + Expected_Log_Rs(RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, i), sigma=pm.math.exp(log_infected_noise))
        #     log_infecteds.append(log_infected)
        expanded_r_walk_noise = pt.tile(
                10 * pt.extra_ops.cumsum(log_infected_noise, axis=-1),
                1,
            )[:nRs, :data['obs'].shape[1]]
        

        growth = pm.Deterministic("growth", RegionR + 
                                  ActiveCMs_NPIs@CM_alpha + Wearing_alpha*ActiveCMs_wearing + 
                                  Mobility_alpha*ActiveCMs_mobility)# + expanded_r_walk_noise)
        
        log_infected = pm.Deterministic(
                "log_infected",
                pt.reshape(InitialSize_log, (nRs, 1))
                + growth.cumsum(axis=1),
            )
        

        obs = pm.NegativeBinomial('obs', alpha=pt.exp(pt.reshape(psi, (nRs, 1))), mu=pt.exp(log_infected), observed=true_obs)
        
        # obs = pm.Bernoulli("obs", logit_p=logits, shape=(num_actors, num_blocks, num_repeats_plate), observed=true_obs)
            
    def joint_logdensity(params, data, covariates):
        #Global
        CM_alpha = stats.norm.logpdf(params.CM_alpha, 0., cm_prior_scale).sum()
        Wearing_alpha = stats.norm.logpdf(params.Wearing_alpha, wearing_mean, wearing_sigma).sum()
        Mobility_alpha = stats.norm.logpdf(params.Mobility_alpha, mobility_mean, mobility_sigma).sum()
        RegionR = stats.norm.logpdf(params.RegionR, R_prior_mean_mean, R_prior_mean_scale + R_noise_scale).sum()
        InitialSize_log_mean = stats.norm.logpdf(params.InitialSize_log_mean, math.log(1000), 0.5).sum()
        log_infected_noise_mean = stats.norm.logpdf(params.log_infected_noise_mean, 0, 0.25).sum()
        
        #Region level
        InitialSize_log = stats.norm.logpdf(params.InitialSize_log, params.InitialSize_log_mean, 0.5).sum()
        log_infected_noise = stats.norm.logpdf(params.log_infected_noise, 0, 0.25).sum()
        
        #Day level
        psi = stats.norm.logpdf(params.psi, 0., 1).sum()
        
        Expected_Log_Rs = RegionR + CM_alpha@ActiveCMs_NPIs + Wearing_alpha*ActiveCMs_wearing + Mobility_alpha*ActiveCMs_mobility 
        
        log_infecteds = [InitialSize_log]
        
        for i in range(data['obs'].shape[1]):
            log_infected = stats.norm.logpdf(params[f'log_infected_{i}'], log_infecteds[i] + Expected_Log_Rs[i], params.log_infected_noise).sum()
            log_infecteds.append(log_infected)
        
        obs = stats.nbinom.logpmf(data, jnp.exp(params.psi), jax.nn.sigmoid(params.log_infected)).sum()
        
        return CM_alpha + Wearing_alpha + Mobility_alpha + RegionR + InitialSize_log_mean + log_infected_noise_mean + InitialSize_log + log_infected_noise + psi + obs
    
    def init_param_fn(seed):
        """
        initialize a, b & thetas
        """
        key1, key2, key3, key4, key5, key6, key7, key8, key9 = jax.random.split(seed, 9)
        log_infected_keys = jax.random.split(key8, data['obs'].shape[1])
        return params(
            CM_alpha = jax.random.normal(key1, shape=(nCMs-2,)),
            Wearing_alpha = jax.random.normal(key2),
            Mobility_alpha = jax.random.normal(key3),
            RegionR = jax.random.normal(key4),
            InitialSize_log_mean = jax.random.normal(key5),
            log_infected_noise_mean = jax.random.normal(key6),
            InitialSize_log = jax.random.normal(key7, shape=(nRs,)),
            log_infected_noise = jax.random.normal(key8, shape=(nRs)),
            psi = jax.random.normal(key9, shape=(nRs,data['obs'].shape[1])),
            for i in range(data['obs'].shape[1]):
                f'log_infected_{i}': jax.random.normal(log_infected_keys[i], shape=(nRs,data['obs'].shape[1]))
        )
        
        
        
        
        
        

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