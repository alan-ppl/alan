import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries, Data, Group, QEMParam, Marginals, checkpoint, mean, mean2
from functorch.dim import Dim 
import sys
import matplotlib.pyplot as plt
from alan.Plate import update_scope

DATA_SEED = 100
USE_MISSPECIFIED_PRIOR = False
USE_MISSPECIFIED_Q = False

PRINT_QEM_MEANS = False

USE_SMOOTHING = False # currently via FFBS variant (slow!)

def get_gaussians_from_moments_dict(mean, mean2, min_var=None):
    assert mean.keys() == mean2.keys()

    dists = {}
    for key in mean.keys():
        var = (mean2[key] - mean[key]**2).sqrt()
        if min_var is not None:
            var = t.maximum(var, min_var)
            var[var.isnan()] = min_var
        dists[key] = t.distributions.Normal(mean[key], var)
        
    return dists

def get_P(wrong_init=False):
    P = Plate( 
        ts_init     = Normal(0. if not wrong_init else 10., 1.),
        ts_log_var  = Normal(0., 1.),

        T = Plate(
            ts = Timeseries('ts_init', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp())),
            a = Normal('ts', 1.)
        ),
    )

    return P

def get_Q(wrong_init=False):
    Q = Plate( 
        ts_init    = Normal(QEMParam(0. if not wrong_init else 5.), QEMParam(1.)),
        ts_log_var = Normal(QEMParam(0.), QEMParam(1.)),

        T = Plate(
            ts = Normal(QEMParam(0.), QEMParam(1.)),
            a = Data(),
        ),
    )

    return Q

def generate_problem(T, data_seed = DATA_SEED, use_misspecified_prior = USE_MISSPECIFIED_PRIOR, use_misspecified_q = USE_MISSPECIFIED_Q):
    t.manual_seed(data_seed)

    P = get_P()
    Q = get_Q(wrong_init=use_misspecified_q)

    bP = BoundPlate(P, {'T': T})
    bQ = BoundPlate(Q, {'T': T})

    true_latents = bP.sample()

    data = {'a': true_latents.pop('a')}

    P = get_P(wrong_init=use_misspecified_prior)
    bP = BoundPlate(P, {'T': T})

    problem = Problem(bP, bQ, data)

    return problem, true_latents, data

def get_true_posterior(T, true_latents, data):
    prior_cov = t.zeros(T, T)
    diag_var = 1.
    for i in range(T):
        diag_var = diag_var*0.9**2 + true_latents['ts_log_var'].exp()**2

        prior_cov[i, i] = diag_var
        future_covs = diag_var * 0.9**t.arange(T-i)
        prior_cov[i, i:] = future_covs
        prior_cov[i:, i] = future_covs

    true_dist = t.distributions.MultivariateNormal(t.zeros(T), prior_cov.rename(None) + t.eye(T))
    # known_elbo = true_dist.log_prob(data['a'].rename(None))

    like_prec = t.eye(T)
    prior_prec = t.inverse(prior_cov)
    post_prec = prior_prec + like_prec
    post_cov = t.inverse(post_prec)
    post_mean = post_cov @ like_prec @ data['a']

    true_posterior = t.distributions.MultivariateNormal(post_mean, post_cov)

    # print('True posterior mean:', post_mean)
    
    return true_posterior, post_mean, post_cov

 # For each iteration we want to save the QEM parameters to a dictionary
def update_param_dict(param_dict, new_params, i):
    for key, value in new_params.items():
        param_dict[key][i, ...] = value.clone()
        
def get_sample_and_dims_for_particle_filter(problem, T, K, K_particles, seed, init_P, init_Q, log_var_P, log_var_Q, smc_elbos, i, PRINT_QEM_MEANS=False):
    t.manual_seed(seed)
    sample = problem.sample(K=K)
    smc_elbos[i] = sample.elbo_nograd().item()

    if PRINT_QEM_MEANS:
        print(f"{i} SMC ts_init: {problem.Q._qem_means.ts_init_mean} {problem.Q._qem_means.ts_init_mean2}")
        print(f"{i} SMC ts_log_var: {problem.Q._qem_means.ts_log_var_mean} {problem.Q._qem_means.ts_log_var_mean2}")
                
    # samples of parameters in P that the timeseries depends on
    ts_init = sample.detached_sample['ts_init']
    ts_log_var = sample.detached_sample['ts_log_var']

    # timeseries samples
    ts = sample.detached_sample['T']['ts']

    # organise dims
    T_dim = ts.dims[1]
    assert T_dim.size == T

    K_dim = Dim('K', K_particles)

    init_dim = ts_init.dims[0]
    log_var_dim = ts_log_var.dims[0]
    ts_param_dims = [init_dim, log_var_dim]

    # get importance weights (p/q) for the K dims of hyperprior samples (ts_init and ts_log_var) and their use in the timeseries
    lpq = [
        init_P.log_prob(ts_init.order(init_dim)) - init_Q.log_prob(ts_init.order(init_dim)) - t.tensor(K).log(),
        log_var_P.log_prob(ts_log_var.order(log_var_dim)) - log_var_Q.log_prob(ts_log_var.order(log_var_dim)) - t.tensor(K).log(),
    ]

    lpq[0] = lpq[0][init_dim]
    lpq[1] = lpq[1][log_var_dim]
    
    return ts_init, ts_log_var, ts, lpq, T_dim, K_dim, init_dim, log_var_dim, ts_param_dims


def smc_qem_update(smc_qem_weights, problem, lr, means, means2, ts_mean, ts_mean2, ts_param_dims, ts_init, ts_log_var):
    # exponentiate and normalise the weights
    for key, val in smc_qem_weights.items():
        smc_qem_weights[key] = (val - val.amax(ts_param_dims)).exp()
        smc_qem_weights[key] /= smc_qem_weights[key].sum(ts_param_dims)

    # first, update ts_init
    new_moment1 = (ts_init    * smc_qem_weights['ts_init']).sum(ts_param_dims) 
    new_moment2 = (ts_init**2 * smc_qem_weights['ts_init']).sum(ts_param_dims) 

    problem.Q._qem_means.ts_init_mean.mul_(1-lr).add_(new_moment1, alpha=lr)
    problem.Q._qem_means.ts_init_mean2.mul_(1-lr).add_(new_moment2, alpha=lr)

    means['ts_init'] = means['ts_init'] * (1-lr) + new_moment1 * lr
    means2['ts_init'] = means2['ts_init'] * (1-lr) + new_moment2 * lr

    # then, ts_log_var
    new_moment1 = (ts_log_var    * smc_qem_weights['ts_log_var']).sum(ts_param_dims) 
    new_moment2 = (ts_log_var**2 * smc_qem_weights['ts_log_var']).sum(ts_param_dims) 

    problem.Q._qem_means.ts_log_var_mean.mul_(1-lr).add_(new_moment1, alpha=lr)
    problem.Q._qem_means.ts_log_var_mean2.mul_(1-lr).add_(new_moment2, alpha=lr)

    means['ts_log_var'] = means['ts_log_var'] * (1-lr) + new_moment1 * lr
    means2['ts_log_var'] = means2['ts_log_var'] * (1-lr) + new_moment2 * lr
    
    # finally, the ts latents
    ts_mean = (ts_mean * smc_qem_weights['ts']).sum(ts_param_dims) 
    ts_mean2 = (ts_mean2 * smc_qem_weights['ts']).sum(ts_param_dims) 

    problem.Q._qem_means.ts_mean.mul_(1-lr).add_(ts_mean, alpha=lr)
    problem.Q._qem_means.ts_mean2.mul_(1-lr).add_(ts_mean2, alpha=lr)

    means['ts'] = means['ts'] * (1-lr) + ts_mean * lr
    means2['ts'] = means2['ts'] * (1-lr) + ts_mean2 * lr

    # update the conventional parameters
    problem.Q._update_qem_convparams()

    updated_dists = get_gaussians_from_moments_dict(means, means2)
    init_Q = updated_dists['ts_init']
    log_var_Q = updated_dists['ts_log_var']
    ts_Q = updated_dists['ts']
    
    return init_Q, log_var_Q, ts_Q

def run_smc_qem_update(T, K, K_particles, i, transition, emission, use_smoothing, ess, data, ts_init, ts_log_var, lpq, T_dim, K_dim, init_dim, log_var_dim, ts_param_dims):
    # set up the particle filter
    particles = t.zeros(K_particles, T, K, K)[K_dim, T_dim, init_dim, log_var_dim]
    marginal_ll = t.zeros((K, K))[ts_param_dims]

    if use_smoothing:
        no_resample_log_weights = t.zeros(K_particles, K, K, T)[K_dim, init_dim, log_var_dim]
        no_resample_particles = t.zeros(K_particles, K, K, T)[K_dim, init_dim, log_var_dim]

    ts_mean = t.zeros(K, K, T)[init_dim, log_var_dim]
    ts_mean2 = t.zeros(K, K, T)[init_dim, log_var_dim]

    for timestep in range(T):
        # create scope to pass to the transition distribution
        scope = {'prev': particles.order(T_dim)[timestep-1] if timestep > 0 else ts_init, 
                'ts_log_var': ts_log_var}
        
        # each of our K x K particle filters needs K_particles 
        scope = {key: val.expand(K_particles)[K_dim] for key, val in scope.items()} 
        
        # sample K_particles xs from the transition
        xs = transition.trans.sample(scope, reparam=False, active_platedims=ts_param_dims, K_dim=K_dim) 
        particles = particles.order(T_dim)
        particles[timestep] = xs
        particles = particles[T_dim]

        # calculate the likelihood of the data given the particles
        log_weights, _ = emission.finalize('a').log_prob(data['a'][timestep], {'ts': particles.order(T_dim)[timestep]}, T_dim, K_dim)
        
        if use_smoothing:
            # keep track of all particles and weights at each timestep for smoothing (i.e. don't overwrite historical samples with resampled particles)
            no_resample_particles[timestep] = xs
            no_resample_log_weights[timestep] = log_weights

        # add to the the marginal log likelihood
        marginal_ll += t.logsumexp(log_weights.order(K_dim), 0) - t.tensor(K_particles).log() # shape [K, K]

        # shift up log_weights and exponentiate
        weights = (log_weights - log_weights.amax(K_dim)).exp() # shape [K_particles, K, K]

        # calculate the mean of the particles for this timestep (for each of the K x K particle filters we're running)
        ts_mean[timestep]  = ((particles.order(T_dim)[timestep]    * weights).sum(K_dim)) / (weights.sum(K_dim)) # shape [K, K]
        ts_mean2[timestep] = ((particles.order(T_dim)[timestep]**2 * weights).sum(K_dim)) / (weights.sum(K_dim)) # shape [K, K]

        # resample the particles (TODO: ONLY RESAMPLE IF ESS IS LOW e.g. < K_particles*0.6ish)
        resampled_indices = t.multinomial(weights.order(K_dim), K_particles, replacement=True)  # shape [K_particles, K, K]
        
        particles = particles.order(K_dim).gather(0, resampled_indices)[K_dim]  # shape [T, K_particles, K, K]

    if use_smoothing:
        # calculate the smoothing weights
        smoothing_weights = t.zeros(K_particles, K, K, T)[K_dim, init_dim, log_var_dim]

        K_dim2 = Dim('K2', K_particles)
        pairwise_smoothing_log_weights = t.zeros(K_particles, K_particles, K, K, T-1)[K_dim, K_dim2, init_dim, log_var_dim]

        no_resample_log_weights = (no_resample_log_weights.exp() / no_resample_log_weights.exp().sum(K_dim)).log()

        Xs = no_resample_particles
        Ws = no_resample_log_weights.exp()
        pWs = pairwise_smoothing_log_weights.exp()

        for timestep in range(T-1, -1, -1):
            # print("smoothing timestep", timestep)
            if timestep == T-1:
                smoothing_weights[timestep] = Ws[timestep]
            else:
                smoothing_weights[timestep] = pWs[timestep].sum(K_dim2)

            if timestep > 0:
                for k in range(K_particles):
                    transition_probs = t.zeros(K, K, K_particles)[init_dim, log_var_dim]
                    for l in range(K_particles):                            
                        prev = Xs[timestep-1].order(K_dim)[l].order(*ts_param_dims)
                        dist = t.distributions.Normal(prev, ts_log_var.order(log_var_dim).expand(K_particles).exp())
                        now = Xs[timestep].order(K_dim)[k].order(*ts_param_dims)
                        transition_probs[l] = dist.log_prob(now)[ts_param_dims]  # K x K x K_particles
                        
                        # transition_probs[l] = t.distributions.Normal(Xs[timestep-1].order(K_dim)[l].order(ts_param_dims), ts_log_var.exp()).log_prob(Xs[timestep].order(K_dim)[k].order(ts_param_dims))[ts_param_dims]

                    S_kt = (Ws[timestep-1] * transition_probs[K_dim]).sum(K_dim)  # K x K
                    
                    pWs = pWs.order(K_dim, K_dim2)
                    pWs[l, k, timestep-1] = smoothing_weights[timestep].order(K_dim)[k] * Ws[timestep-1].order(K_dim)[k] * transition_probs[k] / S_kt
                    pWs = pWs[K_dim2, K_dim]

        smooth_ts_mean = (no_resample_particles * smoothing_weights).sum(K_dim) / (smoothing_weights.sum(K_dim))
        smooth_ts_mean2 = (no_resample_particles**2 * smoothing_weights).sum(K_dim) / (smoothing_weights.sum(K_dim))
                    

    # calculate the ESS
    ess[..., i] = 1/(((log_weights.exp()/log_weights.exp().sum(K_dim))**2).sum(K_dim)).order(init_dim, log_var_dim)

    # now we need to use the marginal_lpq to calculate moments of each parameter
    smc_qem_weights = {'ts_init':    marginal_ll + lpq[0],
                    'ts_log_var': marginal_ll + lpq[1],
                    'ts':         marginal_ll}
                    #    'ts':         marginal_ll + lpq_alan[2]}
            
    if use_smoothing:
        ts_mean = smooth_ts_mean 
        ts_mean2 = smooth_ts_mean2
        
    return smc_qem_weights, ts_mean, ts_mean2


def run(data_seed = DATA_SEED, use_misspecified_prior = USE_MISSPECIFIED_PRIOR, use_misspecified_q = USE_MISSPECIFIED_Q, use_smoothing = USE_SMOOTHING, plot_folder=''):
    T = 5

    K = 25
    K_particles = K
    lr = 0.1

    num_iters = 50

    # Figure out the true posterior
    problem, true_latents, data = generate_problem(T=T, data_seed=data_seed, use_misspecified_prior=use_misspecified_prior, use_misspecified_q=use_misspecified_q)
    true_posterior, post_mean, post_cov = get_true_posterior(T, true_latents, data)

    # REGULAR QEM INFERENCE (where timeseries observations are independent (conditioned on ts_init and ts_log_var))
    problem, true_latents, data = generate_problem(T=T, data_seed=data_seed, use_misspecified_prior=use_misspecified_prior, use_misspecified_q=use_misspecified_q)

    indep_elbos = t.zeros(num_iters)
    indep_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
    update_param_dict(indep_params, problem.Q.qem_params(), 0)

    for i in range(num_iters):
        print(f"{i} QEM")
        t.manual_seed(i)
        sample = problem.sample(K=K)
        indep_elbos[i] = sample.elbo_nograd().item()

        if PRINT_QEM_MEANS:
            print(f"{i} QEM ts_init: {problem.Q._qem_means.ts_init_mean} {problem.Q._qem_means.ts_init_mean2}")
            print(f"{i} QEM ts_log_var: {problem.Q._qem_means.ts_log_var_mean} {problem.Q._qem_means.ts_log_var_mean2}")

        sample.update_qem_params(lr)
        update_param_dict(indep_params, problem.Q.qem_params(), i)


    # SMC QEM INFERENCE (where we use a particle filter to infer the latent states and the model evidence, exploiting the temporal dependencies of the timeseries)
    smc_elbos_no_smoothing = None
    smc_params_no_smoothing = None
    
    smc_elbos_smoothing = None
    smc_params_smoothing = None
    
    ess_no_smoothing = None
    ess_smoothing = None
    
    SMC_successful_iters = num_iters
    
    for use_smoothing in [False, True]:
        problem, true_latents, data = generate_problem(T=T, data_seed=data_seed, use_misspecified_prior=use_misspecified_prior, use_misspecified_q=use_misspecified_q)

        smc_elbos = t.zeros(num_iters)
        smc_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
        update_param_dict(smc_params, problem.Q.qem_params(), 0)

        ess = t.zeros(K, K, num_iters)

        # define the transition and emission distributions (we're using alan distributions for simplicity when dealing with torchdims)
        transition = Timeseries('ts_init', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp()))
        emission = Normal('ts', 1.)
        
        # define the approximate posterior distributions
        init_Q = t.distributions.Normal(0., 1.)
        log_var_Q = t.distributions.Normal(0., 1.)
        # ts_Q = t.distributions.Normal(t.zeros(T), t.ones(T))

        # define the prior distributions
        init_P = t.distributions.Normal(0., 1.)
        log_var_P = t.distributions.Normal(0., 1.)

        # for each of the above three distributions, we need to keep track of the first and second moments
        means = {'ts_init': 0., 'ts_log_var': 0., 'ts': t.zeros(T)}
        means2 = {'ts_init': 1., 'ts_log_var': 1., 'ts': t.ones(T)}

        try:
            for i in range(num_iters):
                print(f"{i} SMC")
                
                # get sample for the particle filter
                ts_init, ts_log_var, ts, lpq, T_dim, K_dim, init_dim, log_var_dim, ts_param_dims = get_sample_and_dims_for_particle_filter(problem, T, K, K_particles, i, init_P, init_Q, log_var_P, log_var_Q, smc_elbos, i, PRINT_QEM_MEANS)
                
                # run the particle filter
                smc_qem_weights, ts_mean, ts_mean2 = run_smc_qem_update(T, K, K_particles, i, transition, emission, use_smoothing, ess, data, ts_init, ts_log_var, lpq, T_dim, K_dim, init_dim, log_var_dim, ts_param_dims)

                # update the QEM parameters
                init_Q, log_var_Q, ts_Q = smc_qem_update(smc_qem_weights, problem, lr, means, means2, ts_mean, ts_mean2, ts_param_dims, ts_init, ts_log_var)
            
                # save the parameters for this iteration
                update_param_dict(smc_params, problem.Q.qem_params(), i)

        except Exception as e:
            print("SMC failed at iteration", i, "with error:")
            print(e)
            SMC_successful_iters = i

        if use_smoothing:
            smc_elbos_smoothing = smc_elbos
            smc_params_smoothing = smc_params
        else:
            smc_elbos_no_smoothing = smc_elbos
            smc_params_no_smoothing = smc_params
            
    # Extended State Space Inference (use noisy versions of ts_init and ts_log_var in our particle filter, with noise decreasing per timestep, to hopefully infer the true latent states)
    problem, true_latents, data = generate_problem(T=T, data_seed=data_seed, use_misspecified_prior=use_misspecified_prior, use_misspecified_q=use_misspecified_q)

    ess_elbos = t.zeros(num_iters)
    ess_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
    update_param_dict(ess_params, problem.Q.qem_params(), 0)
    
    # K_particles = 500

    ess = t.zeros(K, K, num_iters)

    # define the transition and emission distributions
    transition = lambda prev, ts_log_var: t.distributions.Normal(0.9*prev, ts_log_var.exp())
    emission = lambda ts: t.distributions.Normal(ts, 1.)
    
    def log_prob_timeseries(ts, ts_init, ts_log_var, data, param_transition_sd, prev_init, prev_log_var):
        log_prob = t.zeros(*ts.shape)

        log_prob[:, 0] = t.distributions.Normal(ts_init[:, 0], 1.).log_prob(ts[:, 0]) + t.distributions.Normal(ts[:, 0], 1.).log_prob(data[0])
        for i in range(1, log_prob.shape[1]):
            log_prob[:, i] = t.distributions.Normal(0.9*ts[:, i-1], ts_log_var[:, i].exp()).log_prob(ts[:, i]) + t.distributions.Normal(ts[:, i], 1.).log_prob(data[i])
        
        return log_prob.sum(-1) + t.distributions.Normal(prev_init, param_transition_sd).log_prob(ts_init[:,-1]) + t.distributions.Normal(prev_log_var, param_transition_sd).log_prob(ts_log_var[:,-1])
    
    init_Q = t.distributions.Normal(0., 1.)
    log_var_Q = t.distributions.Normal(0., 1.)
    
    param_transition = lambda prev, sd: t.distributions.Normal(prev, sd)
    
    for i in range(num_iters):
        print(f"{i} ESS")

        particles = t.zeros(K_particles, T)  
        
        ts_init_particles = t.zeros(K_particles, T) 
        ts_log_var_particles = t.zeros(K_particles, T) 
        
        ts_mean = t.zeros(T)
        ts_mean2 = t.zeros(T)
        
        # implement the ESS algorithm
        for timestep in range(T):
            # sample ts_init and ts_log_var from the transition
            param_transition_sd = 1.#/(i+1)
            
            prev_init = ts_init_particles[:, timestep-1] if timestep > 0 else t.ones(K_particles) * init_Q.sample()
            prev_log_var = ts_log_var_particles[:, timestep-1] if timestep > 0 else t.ones(K_particles) * log_var_Q.sample()
                        
            ts_init_particles[:, timestep] = param_transition(prev_init, param_transition_sd).sample()
            ts_log_var_particles[:, timestep] = param_transition(prev_log_var, param_transition_sd).sample()
            
            # sample K_particles xs from the transition
            prev = particles[:, timestep-1] if timestep > 0 else ts_init_particles[:, 0]
            particles[:, timestep] = transition(prev, ts_log_var_particles[:, timestep]).sample()
                        
            # calculate the likelihood of the data given the particles
            log_weights = log_prob_timeseries(particles[:,:timestep+1], ts_init_particles[:, :timestep+1], ts_log_var_particles[:, :timestep+1], data['a'][:timestep+1], param_transition_sd, prev_init, prev_log_var)
            log_weights[log_weights.isnan()] = -t.inf

            # add to the the marginal log likelihood
            # marginal_ll += t.logsumexp(log_weights, 0) - t.tensor(K_particles).log() # shape [K, K]

            # shift up log_weights and exponentiate
            weights = (log_weights - log_weights.amax(0)).exp() # shape [K_particles, K, K]
            
            # resample the particles (TODO: ONLY RESAMPLE IF ESS IS LOW e.g. < K_particles*0.6ish)
            resampled_indices = t.multinomial(weights, K_particles, replacement=True)  # shape [K_particles, K, K]
                        
            # particles = particles.gather(0, resampled_indices.expand(T))  # shape [T, K_particles, K, K]
            particles = particles[resampled_indices]
            
            # use the resampled particles to update the ts_init and ts_log_var particles
            ts_init_particles = ts_init_particles[resampled_indices]
            ts_log_var_particles = ts_log_var_particles[resampled_indices]
            
            
            weights /= weights.sum(0)
                        
            # calculate first and second moments of the particles for this timestep
            ts_mean[timestep] = (particles[:,timestep] * weights).sum(0)
            ts_mean2[timestep] = (particles[:,timestep]**2 * weights).sum(0)
            
            
        # calculate the weighted mean of ts_init_particles and ts_log_var_particles for the last timestep
        ts_init_mean = (ts_init_particles[:,-1] * weights).sum(0)
        ts_init_mean2 = (ts_init_particles[:,-1]**2 * weights).sum(0)
        
        ts_log_var_mean = (ts_log_var_particles[:,-1] * weights).sum(0)
        ts_log_var_mean2 = (ts_log_var_particles[:,-1]**2 * weights).sum(0)
        
        # update the Q distributions
        dists = get_gaussians_from_moments_dict({'ts_init': ts_init_mean, 'ts_log_var': ts_log_var_mean, 'ts': ts_mean},
                                                {'ts_init': ts_init_mean2, 'ts_log_var': ts_log_var_mean2, 'ts': ts_mean2}, min_var=t.tensor(1e-6))
        init_Q = dists['ts_init']
        log_var_Q = dists['ts_log_var']
        ts_Q = dists['ts']
        
        # save the parameters for this iteration
        update_param_dict(ess_params, {'ts_init_loc': init_Q.loc, 'ts_init_scale': init_Q.scale,
                                       'ts_log_var_loc': log_var_Q.loc, 'ts_log_var_scale': log_var_Q.scale,
                                       'ts_loc': ts_Q.loc, 'ts_scale': ts_Q.scale}, i)
        
    # PMCMC INFERENCE (where we perform MH MCMC on the parameters using the particle filter to calculate the likelihood)
    problem, true_latents, data = generate_problem(T=T, data_seed=data_seed, use_misspecified_prior=use_misspecified_prior, use_misspecified_q=use_misspecified_q)

    pmcmc_elbos = t.zeros(num_iters)
    pmcmc_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
    update_param_dict(pmcmc_params, problem.Q.qem_params(), 0)
    
    K_particles = 500
    
    # define the transition and emission distributions
    transition = lambda prev, ts_log_var: t.distributions.Normal(0.9*prev, ts_log_var.exp())
    emission = lambda ts: t.distributions.Normal(ts, 1.)
    
    init_Q = t.distributions.Normal(0., 1.)
    log_var_Q = t.distributions.Normal(0., 1.)
    
    param_transition = lambda prev, sd: t.distributions.Normal(prev, sd)
    
    ts_inits = t.zeros(num_iters)
    ts_log_vars = t.zeros(num_iters)
    
    ts_inits[0] = param_transition(0,1).sample()
    ts_log_vars[0] = param_transition(0,1).sample()
    
    marginal_ll = t.zeros(num_iters)

    for i in range(num_iters):
        print(f"{i} PMCMC")
        particles = t.zeros(K_particles, T)  
        
        ts_mean = t.zeros(T)
        ts_mean2 = t.zeros(T)
        
        ts_init_proposal = param_transition(ts_inits[i-1], 1).sample()
        ts_log_var_proposal = param_transition(ts_log_vars[i-1], 1).sample()
        
        # implement the particle filter
        for timestep in range(T):
            # sample K_particles xs from the transition
            prev = particles[:, timestep-1] if timestep > 0 else ts_init_proposal.expand(K_particles)
            particles[:, timestep] = transition(prev, ts_log_var_proposal).sample()
                        
            # calculate the likelihood of the data given the particles
            log_weights = emission(particles[:, timestep]).log_prob(data['a'][timestep])

            # add to the the marginal log likelihood
            marginal_ll[i] += t.logsumexp(log_weights, 0) - t.tensor(K_particles).log() # shape [K, K]

            # shift up log_weights and exponentiate
            weights = (log_weights - log_weights.amax(0)).exp() # shape [K_particles, K, K]
            
            # resample the particles (TODO: ONLY RESAMPLE IF ESS IS LOW e.g. < K_particles*0.6ish)
            resampled_indices = t.multinomial(weights, K_particles, replacement=True)  # shape [K_particles, K, K]
                        
            # particles = particles.gather(0, resampled_indices.expand(T))  # shape [T, K_particles, K, K]
            particles = particles[resampled_indices]
            
            weights /= weights.sum(0)
                        
            # calculate first and second moments of the particles for this timestep
            ts_mean[timestep] = (particles[:,timestep] * weights).sum(0)
            ts_mean2[timestep] = (particles[:,timestep]**2 * weights).sum(0)
            
            
        # update the Q distribution for ts
        dists = get_gaussians_from_moments_dict({'ts': ts_mean}, {'ts': ts_mean2}, min_var=t.tensor(1e-6))
        ts_Q = dists['ts']
        
        # do a metroplis-hastings step for ts_init and ts_log_var
        if i > 0:
            alpha = marginal_ll[i] - marginal_ll[i-1] 
            
            alpha += t.distributions.Normal(0, 1).log_prob(ts_init_proposal) - t.distributions.Normal(0, 1).log_prob(ts_inits[i-1])
            alpha += param_transition(ts_init_proposal, 1).log_prob(ts_inits[i-1]) - param_transition(ts_inits[i-1], 1).log_prob(ts_init_proposal)
            
            alpha += t.distributions.Normal(0, 1).log_prob(ts_log_var_proposal) - t.distributions.Normal(0, 1).log_prob(ts_log_vars[i-1])
            alpha += param_transition(ts_log_var_proposal, 1).log_prob(ts_log_vars[i-1]) - param_transition(ts_log_vars[i-1], 1).log_prob(ts_log_var_proposal)
            
            alpha = min(alpha, t.tensor(0.))
            print(alpha)
            if t.rand(1) < alpha.exp():
                print("accepted")
                ts_inits[i] = ts_init_proposal
                ts_log_vars[i] = ts_log_var_proposal
            else:
                ts_inits[i] = ts_inits[i-1]
                ts_log_vars[i] = ts_log_vars[i-1]
        
        
        # save the parameters for this iteration
        update_param_dict(pmcmc_params, {'ts_init_loc': ts_inits[i], 'ts_init_scale': t.zeros(()),
                                       'ts_log_var_loc': ts_log_vars[i], 'ts_log_var_scale': t.zeros(()),
                                       'ts_loc': ts_Q.loc, 'ts_scale': ts_Q.scale}, i)
        

        
    # calculate the weighted mean of ts_init_particles and ts_log_var_particles for the last timestep
    # ts_init_mean = (ts_init_particles[:,-1] * weights).sum(0)
    # ts_init_mean2 = (ts_init_particles[:,-1]**2 * weights).sum(0)
    
    # ts_log_var_mean = (ts_log_var_particles[:,-1] * weights).sum(0)
    # ts_log_var_mean2 = (ts_log_var_particles[:,-1]**2 * weights).sum(0)


    # PLOT RESULTS
    # PLOT 1: ELBOs
    plt.plot(indep_elbos, label='independent inference on timeseries', color='blue')
    plt.plot(smc_elbos_no_smoothing, label='filtered inference on timeseries', color='green')
    plt.plot(smc_elbos_smoothing, label='smoothed inference on timeseries', color='orange')
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.xlim(0, SMC_successful_iters-1)
    plt.legend()
    plt.savefig(f'{plot_folder}elbos{"_wrong_prior" if USE_MISSPECIFIED_PRIOR else ""}.png')
    plt.close()

    # PLOT 2: QEM PARAM VALUES
    fig, axs = plt.subplots(1, len(true_latents) + T-1, figsize=((2+T)*3, 3))
    col = 0
    for i, key in enumerate(true_latents.keys()):
        _indep_vals = indep_params[key+'_loc'].rename(None)
        _indep_errs = indep_params[key+'_scale'].rename(None)

        _smc_vals = smc_params_no_smoothing[key+'_loc'].rename(None)
        _smc_errs = smc_params_no_smoothing[key+'_scale'].rename(None)
        
        _smc_vals_smoothed = smc_params_smoothing[key+'_loc'].rename(None)
        _smc_errs_smoothed = smc_params_smoothing[key+'_scale'].rename(None)
        
        _ess_vals = ess_params[key+'_loc'].rename(None)
        _ess_errs = ess_params[key+'_scale'].rename(None)
        
        _pmcmc_vals = pmcmc_params[key+'_loc'].rename(None)
        _pmcmc_errs = pmcmc_params[key+'_scale'].rename(None)
        if key == 'ts':
            for j in range(T):
                indep_vals = _indep_vals[:, j]
                indep_errs = _indep_errs[:, j]

                smc_vals = _smc_vals[:, j]
                smc_errs = _smc_errs[:, j]
                
                smc_vals_smoothed = _smc_vals_smoothed[:, j]
                smc_errs_smoothed = _smc_errs_smoothed[:, j]
                
                ess_vals = _ess_vals[:, j]
                ess_errs = _ess_errs[:, j]
                
                pmcmc_vals = _pmcmc_vals[:, j]
                pmcmc_errs = _pmcmc_errs[:, j]

                axs[col].plot([true_latents[key][j].rename(None)]*num_iters, label='true', linestyle='--', color='black')

                axs[col].plot([post_mean[j].rename(None)]*num_iters, label='posterior mean', linestyle='--', color='red')

                axs[col].plot(indep_vals, label='independent', color='blue')
                axs[col].fill_between(t.arange(num_iters), indep_vals-indep_errs, indep_vals+indep_errs, alpha=0.2, color='blue')

                axs[col].plot(smc_vals, label='smc filtered', color='green')
                axs[col].fill_between(t.arange(num_iters), smc_vals - smc_errs, smc_vals + smc_errs, alpha=0.2, color='green')
                
                axs[col].plot(smc_vals_smoothed, label='smc smoothed', color='orange')
                axs[col].fill_between(t.arange(num_iters), smc_vals_smoothed - smc_errs_smoothed, smc_vals_smoothed + smc_errs_smoothed, alpha=0.2, color='orange')
                
                axs[col].plot(ess_vals, label='ess', color='purple')
                axs[col].fill_between(t.arange(num_iters), ess_vals - ess_errs, ess_vals + ess_errs, alpha=0.2, color='purple')
                
                axs[col].plot(pmcmc_vals, label='pmcmc', color='brown')
                axs[col].fill_between(t.arange(num_iters), pmcmc_vals - pmcmc_errs, pmcmc_vals + pmcmc_errs, alpha=0.2, color='brown')

                axs[col].set_title(key + ' at timestep ' + str(j+1))
                # axs[col].legend()
                col += 1
        else:
            indep_vals = _indep_vals
            indep_errs = _indep_errs
            
            smc_vals = _smc_vals
            smc_errs = _smc_errs
            
            smc_vals_smoothed = _smc_vals_smoothed
            smc_errs_smoothed = _smc_errs_smoothed
            
            ess_vals = _ess_vals
            ess_errs = _ess_errs
            
            pmcmc_vals = _pmcmc_vals
            pmcmc_errs = _pmcmc_errs

            axs[col].plot([true_latents[key].rename(None)]*num_iters, label='true', linestyle='--', color='black')

            axs[col].plot(indep_vals, label='independent', color='blue')
            axs[col].fill_between(t.arange(num_iters), indep_vals-indep_errs, indep_vals+indep_errs, alpha=0.2, color='blue')

            axs[col].plot(smc_vals, label='smc filtered', color='green')
            axs[col].fill_between(t.arange(num_iters), smc_vals - smc_errs, smc_vals + smc_errs, alpha=0.2, color='green')
            
            axs[col].plot(smc_vals, label='smc smoothed', color='orange')
            axs[col].fill_between(t.arange(num_iters), smc_vals_smoothed - smc_errs_smoothed, smc_vals_smoothed + smc_errs_smoothed, alpha=0.2, color='orange')
            
            axs[col].plot(ess_vals, label='ess', color='purple')
            axs[col].fill_between(t.arange(num_iters), ess_vals - ess_errs, ess_vals + ess_errs, alpha=0.2, color='purple')
            
            axs[col].plot(pmcmc_vals, label='pmcmc', color='brown')
            axs[col].fill_between(t.arange(num_iters), pmcmc_vals - pmcmc_errs, pmcmc_vals + pmcmc_errs, alpha=0.2, color='brown')
            
            axs[col].set_title(key)
            # axs[col].legend()
            col += 1

    for i in range(col):
        axs[i].set_xlim(0, SMC_successful_iters-1)

    axs[0].legend()
    axs[-1].legend()

    fig.tight_layout()
    fig.savefig(f'{plot_folder}errors{"_wrong_prior" if USE_MISSPECIFIED_PRIOR else ""}.png')
    plt.close()


    # PLOT 3: ESSs
    plt.plot(ess.mean(0).transpose(0,1), color='purple')
    plt.plot(ess.mean(1).transpose(0,1), color='orange')
    plt.plot([], label='ts_init', color='purple')
    plt.plot([], label='ts_log_var', color='orange')
    plt.ylabel('(Final) ESS')
    plt.xlabel('Iteration')
    plt.xlim(0, SMC_successful_iters-1)
    plt.ylim(0, K_particles)
    plt.legend()
    plt.savefig(f'{plot_folder}ess.png')
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        run()
    elif len(sys.argv) == 2:
        run(data_seed=int(sys.argv[1]))
    else:
        for i in range(1, len(sys.argv)):
            run(data_seed=int(sys.argv[i]), plot_folder=f'plots/{sys.argv[i]}')
            print(f'Finished run {sys.argv[i]}')
