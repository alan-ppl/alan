import math
import torch as t
from alan_simplified import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, IndependentSample, Normal
from TestProblem import TestProblem

prior_mean = 2
a_scale = 1
b_scale = 1
prior_var = a_scale**2 + b_scale**2
prior_prec = 1/prior_var

like_scale = 3
like_prec = 1/like_scale**2

N = 10
data = 1.5+t.randn(N)
post_prec = prior_prec + data.shape[0]*like_prec
post_mean = (prior_prec*prior_mean + like_prec*data.sum()) / post_prec

marginal_prior_mean = prior_mean*t.ones(N)
marginal_prior_cov = prior_var*t.ones(N, N) + (like_scale**2)*t.eye(N)
known_elbo = t.distributions.MultivariateNormal(marginal_prior_mean, marginal_prior_cov).log_prob(data)


P = Plate(
    a = Normal(prior_mean, a_scale),
    b = Normal('a', b_scale),
    T = Plate(
        d = Normal('b', like_scale),
    ),
)

Q = Plate(
    a = Normal(1, 4),
    b = Normal('a', 1.2),
    T = Plate(
        d = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': N}
data = {'d': data.refine_names('T')}
problem = Problem(P, Q, all_platesizes, data)

moments = [('a', mean), ('a', mean2)]
known_moments = {
    ('a', mean): post_mean,
    ('a', mean): post_mean**2 + 1/post_prec,
}



tp = TestProblem(problem, moments, known_moments=known_moments, known_elbo=known_elbo, moment_K=100, elbo_K=100)

