"""
Plated latent variables, including a non-IID approximate posterior.
"""
import torch as t
from alan import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, Normal, Split
from TestProblem import TestProblem

prior_mean = 2
prior_scale = 2
prior_var = prior_scale**2
prior_prec = 1/prior_var

z_scale = 1.3
d_scale = 1.5

like_var = z_scale**2 + d_scale**2
like_prec = 1/like_var

N = 10
data = 1.5+t.randn(N)
post_prec = prior_prec + data.shape[0]*like_prec
post_mean = (prior_prec*prior_mean + like_prec*data.sum()) / post_prec

marginal_prior_mean = prior_mean*t.ones(N)
marginal_prior_cov = prior_var*t.ones(N, N) + like_var*t.eye(N)
known_elbo = t.distributions.MultivariateNormal(marginal_prior_mean, marginal_prior_cov).log_prob(data)


P = Plate(
    a = Normal(prior_mean, prior_scale),
    T = Plate(
        z = Normal('a', z_scale),
        d = Normal('z', d_scale),
    ),
)

Q = Plate(
    a = Normal(1, 4),
    T = Plate(
        z = Normal(lambda a: 1.5*a, 3.5),
        d = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': N}
data = {'d': data.refine_names('T')}

moments = [('a', mean), ('a', mean2), ('z', mean), ('z', mean2)]
known_moments = {
    ('a', mean): post_mean,
    ('a', mean2): post_mean**2 + 1/post_prec,
}



tp = TestProblem(
    P, Q, all_platesizes, data,
    moments, 
    known_moments=known_moments, 
    known_elbo=known_elbo, 
    moment_K=100, 
    elbo_K=1000, 
    elbo_iters=30,
    elbo_gap_cat=2,
    computation_strategy=Split('T', 3),
)

