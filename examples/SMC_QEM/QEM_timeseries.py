import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries, Data, Group, QEMParam, Marginals, checkpoint, mean, mean2
from functorch.dim import Dim 
import sys
import matplotlib.pyplot as plt
from alan.logpq import lp_getter
from alan.Plate import update_scope

DATA_SEED = 100
USE_MISSPECIFIED_PRIOR = False
USE_MISSPECIFIED_Q = False

PRINT_QEM_MEANS = False

def run(data_seed = DATA_SEED, use_misspecified_prior = USE_MISSPECIFIED_PRIOR, use_misspecified_q = USE_MISSPECIFIED_Q, plot_folder=''):

    T = 5

    K = 25
    K_particles = K
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

    def generate_problem():
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

    # Figure out the true posterior
    problem, true_latents, data = generate_problem()

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

    print('True posterior mean:', post_mean)


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
    problem, true_latents, data = generate_problem()

    smc_elbos = t.zeros(num_iters)
    smc_params = {key : t.zeros((num_iters, *val.shape)) for key, val in problem.Q.qem_params().items()}
    update_param_dict(smc_params, problem.Q.qem_params(), 0)

    ess = t.zeros(K, K, num_iters)

    # define the transition and emission distributions (we're using alan distributions for simplicity when dealing with torchdims)
    transition = Timeseries('ts_init', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp()))
    emission = Normal('ts', 1.)

    SMC_successful_iters = num_iters

    try:
        for i in range(num_iters):
            print(f"{i} SMC")
            t.manual_seed(i)
            sample = problem.sample(K=K)
            smc_elbos[i] = sample.elbo_nograd().item()

            if PRINT_QEM_MEANS:
                print(f"{i} SMC ts_init: {problem.Q._qem_means.ts_init_mean} {problem.Q._qem_means.ts_init_mean2}")
                print(f"{i} SMC ts_log_var: {problem.Q._qem_means.ts_log_var_mean} {problem.Q._qem_means.ts_log_var_mean2}")

            # get importance weights (p/q) for the K dims of hyperprior samples (ts_init and ts_log_var) and the their use in the timeseries
            lpq_scope = {}
            lpq_scope = update_scope(lpq_scope, sample.detached_sample)
            lpq_scope = update_scope(lpq_scope, sample.problem.inputs_params())
       
            lpq, _, _, _ = lp_getter(
                name=None,
                P=sample.P.plate, 
                Q=sample.Q.plate, 
                sample=sample.detached_sample,
                inputs_params=sample.problem.inputs_params(),
                data=sample.problem.data,
                extra_log_factors={'T': {}},
                scope=lpq_scope, 
                active_platedims=[],
                all_platedims=sample.all_platedims,
                groupvarname2Kdim=sample.groupvarname2Kdim,
                varname2groupvarname=sample.problem.Q.varname2groupvarname(),
                sampler=sample.sampler,
                computation_strategy=checkpoint,
                )
                        
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

            # set up the particle filter
            particles = t.zeros(K_particles, T, K, K)[K_dim, T_dim, init_dim, log_var_dim]
            marginal_ll = t.zeros((K, K))[ts_param_dims]

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

                # log_weights = log_weights.order(K_dim).gather(0, resampled_indices)[K_dim]  # shape [K_particles, K, K]

            # calculate the ESS
            ess[..., i] = 1/(((log_weights.exp()/log_weights.exp().sum(K_dim))**2).sum(K_dim)).order(init_dim, log_var_dim)

            # now we need to use the marginal_lpq to calculate moments of each parameter
            smc_qem_weights = {'ts_init':    marginal_ll + lpq[0],
                               'ts_log_var': marginal_ll + lpq[1],
                               'ts':         marginal_ll + lpq[2]}

            # normalise the weights
            for key, val in smc_qem_weights.items():
                smc_qem_weights[key] = (val - val.amax(ts_param_dims)).exp()
                smc_qem_weights[key] /= smc_qem_weights[key].sum(ts_param_dims)

            # first, update ts_init
            new_moment1 = (ts_init    * smc_qem_weights['ts_init']).sum(ts_param_dims) 
            new_moment2 = (ts_init**2 * smc_qem_weights['ts_init']).sum(ts_param_dims) 

            problem.Q._qem_means.ts_init_mean.mul_(1-lr).add_(new_moment1, alpha=lr)
            problem.Q._qem_means.ts_init_mean2.mul_(1-lr).add_(new_moment2, alpha=lr)

            # then, ts_log_var
            new_moment1 = (ts_log_var    * smc_qem_weights['ts_log_var']).sum(ts_param_dims) 
            new_moment2 = (ts_log_var**2 * smc_qem_weights['ts_log_var']).sum(ts_param_dims) 

            problem.Q._qem_means.ts_log_var_mean.mul_(1-lr).add_(new_moment1, alpha=lr)
            problem.Q._qem_means.ts_log_var_mean2.mul_(1-lr).add_(new_moment2, alpha=lr)
            
            # finally, the ts latents
            ts_mean = (ts_mean * smc_qem_weights['ts']).sum(ts_param_dims) 
            ts_mean2 = (ts_mean2 * smc_qem_weights['ts']).sum(ts_param_dims) 

            problem.Q._qem_means.ts_mean.mul_(1-lr).add_(ts_mean, alpha=lr)
            problem.Q._qem_means.ts_mean2.mul_(1-lr).add_(ts_mean2, alpha=lr)

            # update the conventional parameters
            problem.Q._update_qem_convparams()

            # save the parameters for this iteration
            update_param_dict(smc_params, problem.Q.qem_params(), i)

    except Exception as e:
        print("SMC failed at iteration", i, "with error:")
        print(e)
        SMC_successful_iters = i



    # PLOT RESULTS
    # PLOT 1: ELBOs
    plt.plot(indep_elbos, label='independent inference on timeseries', color='blue')
    plt.plot(smc_elbos, label='smc inference on timeseries', color='green')
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

        _smc_vals = smc_params[key+'_loc'].rename(None)
        _smc_errs = smc_params[key+'_scale'].rename(None)
        if key == 'ts':
            for j in range(T):
                indep_vals = _indep_vals[:, j]
                indep_errs = _indep_errs[:, j]

                smc_vals = _smc_vals[:, j]
                smc_errs = _smc_errs[:, j]

                axs[col].plot([true_latents[key][j].rename(None)]*num_iters, label='true', linestyle='--', color='black')

                axs[col].plot([post_mean[j].rename(None)]*num_iters, label='posterior mean', linestyle='--', color='red')

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
