import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries, Data, Group, QEMParam, mean, mean2
from alan import checkpoint
from alan.logpq import lp_getter
from alan.Plate import update_scope
import matplotlib.pyplot as plt

# set manual seed from potentially empty cmd line arg
import sys
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    t.manual_seed(seed)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(device)
K = 10
T = 5
T_extra = 2

ts_log_var = t.zeros(())

def get_P():
    P = Plate( 
        ts_init     = Normal(0., 1.),
        # ts_log_var  = Normal(0., 1.),

        # ts_init2    = Normal(0., 1.),

        T = Plate(
            ts = Timeseries('ts_init', Normal(lambda prev: 0.9*prev,  ts_log_var.exp())),

            # ts = Timeseries('ts_init', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp())),
            # ts2 = Timeseries('ts_init2', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp())),
            a = Normal('ts', 1.)
        ),
    )
    return P

def get_Q():
    Q = Plate( 
        ts_init    = Normal(QEMParam(0.), QEMParam(1.)),
        # ts_log_var = Normal(QEMParam(0.), QEMParam(1.)),

        # ts_init2    = Normal(QEMParam(0.), QEMParam(1.)),

        T = Plate(
            ts = Timeseries('ts_init', Normal(lambda prev: 0.9*prev,  ts_log_var.exp())),
            
            # ts = Timeseries('ts_init', Normal(lambda prev: 0.9*prev, lambda ts_log_var: ts_log_var.exp())),
            # ts2 = Timeseries('ts_init2', Normal(lambda prev: 0.8*prev, lambda ts_log_var: ts_log_var.exp())),

            # ts = Normal(QEMParam(0.), QEMParam(1.)),
            a = Data(),
        ),
    )
    return Q

P = get_P()
Q = get_Q()

bP = BoundPlate(P, {'T': T})
bQ = BoundPlate(Q, {'T': T})

true_latents = bP.sample()

data = {'a': true_latents.pop('a')}

P = get_P()
bP = BoundPlate(P, {'T': T})

problem = Problem(bP, bQ, data)

problem.to(device)
true_latents = {k: v.to(device) for k,v in true_latents.items()}
data = {k: v.to(device) for k,v in data.items()}
ts_log_var = ts_log_var.to(device)


# get true posterior analytically
prior_cov = t.zeros(T, T).to(device)
diag_var = 1.
for i in range(T):
    diag_var = diag_var*0.9**2 + ts_log_var.exp()**2

    prior_cov[i, i] = diag_var
    future_covs = diag_var * 0.9**t.arange(T-i).to(device)
    prior_cov[i, i:] = future_covs
    prior_cov[i:, i] = future_covs

true_dist = t.distributions.MultivariateNormal(t.zeros(T).to(device), prior_cov.rename(None) + t.eye(T).to(device))
# known_elbo = true_dist.log_prob(data['a'].rename(None))

like_prec = t.eye(T).to(device) 
prior_prec = t.inverse(prior_cov)
post_prec = prior_prec + like_prec
post_cov = t.inverse(post_prec)
post_mean = post_cov @ like_prec @ data['a']

true_posterior = t.distributions.MultivariateNormal(post_mean, post_cov)


sample = problem.sample(K=K)
isample = sample.importance_sample(N=4)

print("First importance sample check done.")
# breakpoint()

# see if we can recover the true posterior using importance sampling
max_K = 250

ts_means = t.zeros(max_K, T)

for K in range(1,max_K+1):
    if K % 10 == 0:
        print("K", K)
    sample = problem.sample(K=K)

    # elbo = sample.elbo_vi()

    isample = sample.importance_sample(N=250)

    # get posterior moments from importance samples
    moments = isample.moments([("ts", mean)])

    ts_means[K-1] = moments[0]

ts_means = ts_means.cpu().numpy()
post_mean = post_mean.cpu().numpy()


# plot the results
fig, ax = plt.subplots(1,T, figsize=(5*T,5), sharey=True)

for i in range(T):
    ax[i].plot(ts_means[:,i], label='importance sample')
    ax[i].plot([0, max_K], [post_mean[i], post_mean[i]], 'r--', label='true')

    ax[i].set_title(f"ts{i}")
    ax[i].set_xlabel("K")

ax[0].set_ylabel("posterior mean")
ax[0].legend()

plt.show()
plt.savefig("simple_timeseries.png")

# breakpoint()