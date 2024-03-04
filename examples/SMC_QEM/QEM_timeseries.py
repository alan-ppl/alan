import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries, Data, Group, QEMParam, Marginals, checkpoint
from functorch.dim import Dim 

USE_MISSPECIFIED_PRIOR = False

T = 10

K = 25
lr = 0.1

num_iters = 250

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

def get_Q():
    Q = Plate( 
        ts_init    = Normal(QEMParam(0.), QEMParam(1.)),
        ts_log_var = Normal(QEMParam(0.), QEMParam(1.)),

        T = Plate(
            ts = Normal(QEMParam(0.), QEMParam(1.)),
            a = Data(),
        ),
    )

    return Q

def generate_problem():
    t.manual_seed(0)

    P = get_P()
    Q = get_Q()

    bP = BoundPlate(P, {'T': T})
    bQ = BoundPlate(Q, {'T': T})

    true_latents = bP.sample()

    data = {'a': true_latents.pop('a')}

    P = get_P(wrong_init=USE_MISSPECIFIED_PRIOR)
    bP = BoundPlate(P, {'T': T})

    problem = Problem(bP, bQ, data)

    return problem, true_latents, data

# For each iteration we want to save the QEM parameters to a dictionary
def update_param_dict(param_dict, new_params, i):
    for key, value in new_params.items():
        param_dict[key][i, ...] = value.clone()

# REGULAR QEM INFERENCE (where timeseries observations are independent (conditioned on ts_init and ts_log_var))
problem, true_latents, data = generate_problem()

indep_elbos = t.zeros(num_iters)
indep_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
update_param_dict(indep_params, problem.Q.qem_params(), 0)

for i in range(num_iters):
    t.manual_seed(i)
    sample = problem.sample(K=K)
    indep_elbos[i] = sample.elbo_nograd().item()

    sample.update_qem_params(lr)

    update_param_dict(indep_params, problem.Q.qem_params(), i)


# SMC QEM INFERENCE (where we use a particle filter to infer the latent states and the model evidence, exploiting the temporal dependencies of the timeseries)
problem, true_latents, data = generate_problem()

smc_elbos = t.zeros(num_iters)
smc_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
update_param_dict(smc_params, problem.Q.qem_params(), 0)

# define the transition and emission distributions (we're using alan distributions for simplicity when dealing with torchdims)
transition = Timeseries('ts_init', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp()))
emission = Normal('ts', 1.)

for i in range(num_iters):
    t.manual_seed(i)
    sample = problem.sample(K=K)
    smc_elbos[i] = sample.elbo_nograd().item()

    # samples of parameters in P that the timeseries depends on
    ts_init = sample.detached_sample['ts_init']
    ts_log_var = sample.detached_sample['ts_log_var']

    # timeseries samples
    ts = sample.detached_sample['T']['ts']

    # organise dims
    T_dim = ts.dims[1]
    assert T_dim.size == T

    K_dim = Dim('K', K)

    init_dim = ts_init.dims[0]
    log_var_dim = ts_log_var.dims[0]
    ts_param_dims = [init_dim, log_var_dim]

    # set up the particle filter
    particles = t.zeros(K, T, K, K)[K_dim, T_dim, init_dim, log_var_dim]
    marginal_ll = t.zeros((K, K))[ts_param_dims]

    ts_mean = t.zeros(K, K, T)[init_dim, log_var_dim]
    ts_mean2 = t.zeros(K, K, T)[init_dim, log_var_dim]

    for timestep in range(T):
        # create scope to pass to the transition distribution
        scope = {'prev': particles.order(T_dim)[timestep-1] if timestep > 0 else ts_init, 
                 'ts_log_var': ts_log_var}
        
        # each of our K x K particle filters needs K particles
        scope = {key: val.expand(K)[K_dim] for key, val in scope.items()} 
        
        # sample K particles xs from the transition
        xs = transition.trans.sample(scope, reparam=False, active_platedims=ts_param_dims, K_dim=K_dim) 
        particles = particles.order(T_dim)
        particles[timestep] = xs
        particles = particles[T_dim]

        # calculate the likelihood of the data given the particles
        log_weights, _ = emission.finalize('a').log_prob(data['a'][timestep], {'ts': particles.order(T_dim)[timestep]}, T_dim, K_dim)

        # add to the marginal likelihood
        marginal_ll += t.logsumexp(log_weights.order(K_dim), 0) - t.tensor(K).log() # shape [K, K]

        # shift up log_weights and exponentiate
        weights = (log_weights - log_weights.amax([K_dim, init_dim, log_var_dim])).exp() # shape [K, K, K]

        # calculate the mean of the particles for this timestep (for each of the K x K particle filters we're running)
        ts_mean[timestep]  = ((particles.order(T_dim)[timestep]    * weights).mean(K_dim)) / (weights.mean(K_dim)) # shape [K, K]
        ts_mean2[timestep] = ((particles.order(T_dim)[timestep]**2 * weights).mean(K_dim)) / (weights.mean(K_dim)) # shape [K, K]

        # resample the particles
        resampled_indices = t.multinomial(weights, K, replacement=True)  # shape [K, K]
        particles = particles.order(K_dim).gather(0, resampled_indices)  # shape [T, K, K, K]

    # shift up marginal log likelihood and exponentiate (per parameter combination of ts_init and ts_log_var)
    marginal_ll = (marginal_ll - marginal_ll.amax(ts_param_dims)).exp()

    # now we need to use the marginal_ll to calculate moments of each parameter
    # first, ts_init
    new_moment1 = (sample.detached_sample['ts_init']    * marginal_ll.sum(log_var_dim)).sum(init_dim) / marginal_ll.sum(ts_param_dims)
    new_moment2 = (sample.detached_sample['ts_init']**2 * marginal_ll.sum(log_var_dim)).sum(init_dim) / marginal_ll.sum(ts_param_dims)

    problem.Q._qem_means.ts_init_mean.mul_(1-lr).add_(new_moment1, alpha=lr)
    problem.Q._qem_means.ts_init_mean2.mul_(1-lr).add_(new_moment2, alpha=lr)

    # then, ts_log_var
    new_moment1 = (sample.detached_sample['ts_log_var']    * marginal_ll.sum(init_dim)).sum(log_var_dim) / marginal_ll.sum(ts_param_dims)
    new_moment2 = (sample.detached_sample['ts_log_var']**2 * marginal_ll.sum(init_dim)).sum(log_var_dim) / marginal_ll.sum(ts_param_dims)

    problem.Q._qem_means.ts_log_var_mean.mul_(1-lr).add_(new_moment1, alpha=lr)
    problem.Q._qem_means.ts_log_var_mean2.mul_(1-lr).add_(new_moment2, alpha=lr)
    
    # finally, the ts latents
    ts_mean = (ts_mean * marginal_ll).sum(ts_param_dims) / marginal_ll.sum(ts_param_dims)
    ts_mean2 = (ts_mean2 * marginal_ll).sum(ts_param_dims) / marginal_ll.sum(ts_param_dims)
    # ts_mean2 = (ts_mean**2 * marginal_ll).sum(ts_param_dims) / marginal_ll.sum(ts_param_dims)

    problem.Q._qem_means.ts_mean.mul_(1-lr).add_(ts_mean, alpha=lr)
    problem.Q._qem_means.ts_mean2.mul_(1-lr).add_(ts_mean2, alpha=lr)

    # breakpoint()
    # update the conventional parameters
    problem.Q._update_qem_convparams()

    # save the parameters for this iteration
    update_param_dict(smc_params, problem.Q.qem_params(), i)


# PLOT RESULTS
import matplotlib.pyplot as plt

# PLOT 1: ELBOs
plt.plot(indep_elbos, label='independent inference on timeseries')
plt.plot(smc_elbos, label='smc inference on timeseries')
plt.ylabel('ELBO')
plt.xlabel('Iteration')
plt.legend()
plt.savefig(f'elbos{"_wrong_prior" if USE_MISSPECIFIED_PRIOR else ""}.png')
plt.close()

# PLOT 2: QEM PARAM VALUES
fig, axs = plt.subplots(1, len(true_latents) + T-1, figsize=((2+T)*3, 3))
col = 0
for i, key in enumerate(true_latents.keys()):
    _indep_vals = indep_params[key+'_loc'].rename(None)
    _indep_errs = indep_params[key+'_scale'].rename(None)

    _smc_vals = smc_params[key+'_loc'].rename(None)
    _smc_errs = smc_params[key+'_scale'].rename(None)
    if key == 'ts':
        for j in range(T):
            indep_vals = _indep_vals[:, j]
            indep_errs = _indep_errs[:, j]

            smc_vals = _smc_vals[:, j]
            smc_errs = _smc_errs[:, j]

            axs[col].plot([true_latents[key][j].rename(None)]*num_iters, label='true', linestyle='--', color='black')

            axs[col].plot(indep_vals, label='independent', color='blue')
            axs[col].fill_between(t.arange(num_iters), indep_vals-indep_errs, indep_vals+indep_errs, alpha=0.2, color='blue')

            axs[col].plot(smc_vals, label='smc', color='green')
            axs[col].fill_between(t.arange(num_iters), smc_vals - smc_errs, smc_vals + smc_errs, alpha=0.2, color='green')
            
            axs[col].set_title(key + ' at timestep ' + str(j+1))
            # axs[col].legend()
            col += 1
    else:
        indep_vals = _indep_vals
        indep_errs = _indep_errs
        
        smc_vals = _smc_vals
        smc_errs = _smc_errs

        axs[col].plot([true_latents[key].rename(None)]*num_iters, label='true', linestyle='--', color='black')

        axs[col].plot(indep_vals, label='independent', color='blue')
        axs[col].fill_between(t.arange(num_iters), indep_vals-indep_errs, indep_vals+indep_errs, alpha=0.2, color='blue')

        axs[col].plot(smc_vals, label='smc', color='green')
        axs[col].fill_between(t.arange(num_iters), smc_vals - smc_errs, smc_vals + smc_errs, alpha=0.2, color='green')
        
        axs[col].set_title(key)
        # axs[col].legend()
        col += 1

axs[0].legend()
axs[-1].legend()

fig.tight_layout()
fig.savefig(f'errors{"_wrong_prior" if USE_MISSPECIFIED_PRIOR else ""}.png')
plt.close()
