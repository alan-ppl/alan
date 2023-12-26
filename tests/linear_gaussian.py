"""
Most basic test: univarate Gaussian.  No plated latents, but the data is plated.
"""

import torch as t
from alan import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, Normal
from TestProblem import TestProblem

prior_mean = 2
prior_scale = 2
prior_prec = 1/prior_scale**2

like_scale = 3
like_prec = 1/like_scale**2

mult=2.5

N = 10
data = 1.5+t.randn(N)
post_prec = prior_prec + data.shape[0]*like_prec*mult**2
post_mean = (prior_prec*prior_mean + like_prec*mult**2*(data.sum()/mult)) / post_prec

marginal_prior_mean = prior_mean*mult*t.ones(N)
marginal_prior_cov = ((mult*prior_scale)**2)*t.ones(N, N) + (like_scale**2)*t.eye(N)
known_elbo = t.distributions.MultivariateNormal(marginal_prior_mean, marginal_prior_cov).log_prob(data)


P = Plate(
    a = Normal(prior_mean, prior_scale),
    T = Plate(
        d = Normal(lambda a: mult*a, like_scale),
    ),
)

Q = Plate(
    a = Normal(1, 4),
    T = Plate(
        d = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': N}
data = {'d': data.refine_names('T')}
problem = Problem(P, Q, all_platesizes, data)

known_moments = {
    ('a', mean): post_mean,
    ('a', mean2): post_mean**2 + 1/post_prec,
}
moments = list(known_moments.keys())

tp = TestProblem(problem, moments, known_moments=known_moments, known_elbo=known_elbo, moment_K=10000, elbo_K=10000)
