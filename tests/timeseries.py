import torch as t
from alan import Normal, Timeseries, Plate, BoundPlate, Group, Problem, Data, mean, Split
from TestProblem import TestProblem

T = 4
A = 0.9
init_scale = 1.
timeseries_noise_scale = 0.1
obs_noise_scale = 1.

init_var = init_scale **2
timeseries_noise_var = timeseries_noise_scale**2
obs_noise_var = obs_noise_scale**2

P = Plate(
    init = Normal(0, init_scale),
    T = Plate(
        ts = Timeseries("init", Normal(lambda prev: A*prev, timeseries_noise_scale)),
        obs = Normal('ts', obs_noise_scale),
    ),
)

Q = Plate(
    init = Normal(0, 1),
    T = Plate(
        #ts = Timeseries("init", Normal(lambda prev: A*prev, timeseries_noise_scale)),#Normal(0, 1),
        ts = Normal(0, 1),
        obs = Data(),
    ),
)

t1  = t.arange(T)[:, None]
t2 = t.arange(T)[None, :]

curr_t   = t.min(t1, t2)
future_t = t.max(t1, t2)

prior_cov = t.zeros(T, T)
diag_var = init_var
for i in range(T):
    diag_var = diag_var*A**2 + timeseries_noise_var

    prior_cov[i, i] = diag_var
    future_covs = diag_var * A**t.arange(T-i)
    prior_cov[i, i:] = future_covs
    prior_cov[i:, i] = future_covs

true_dist = t.distributions.MultivariateNormal(t.zeros(T), prior_cov + obs_noise_var * t.eye(T))
data_ts = true_dist.sample()
known_elbo = true_dist.log_prob(data_ts)

like_prec = t.eye(T) / obs_noise_var
prior_prec = t.inverse(prior_cov)
post_prec = prior_prec + like_prec
post_cov = t.inverse(post_prec)
post_mean = post_cov @ like_prec @ data_ts


all_platesizes = {'T': T}
P = BoundPlate(P, all_platesizes)
Q = BoundPlate(Q, all_platesizes)

data = {'obs': data_ts.refine_names('T')}

moments = [('ts', mean)]
known_moments = {
    ('ts', mean): post_mean.refine_names('T')
}

tp = TestProblem(
    P, Q, data, 
    moments, 
    known_moments=known_moments,
    moment_K=1000, 
    elbo_K=1000, 
    known_elbo=known_elbo,
    #computation_strategy=Split('T', 4),
)
