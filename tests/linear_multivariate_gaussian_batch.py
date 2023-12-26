"""
Combines unnamed batch-dimensions with complex multivariate distribution (MultivariateNormal).

Note that the matrix algebra for computing the true posterior mean requires a bit of rearranging,
so e.g. 
prior_mean  is of shape [N, F], and goes into the probabilistic program, while
prior_mean_ is of shape [N, F, 1], and is used to compute the posterior mean
"""

import torch as t
from alan import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, MultivariateNormal
from TestProblem import TestProblem

N = 3
F = 2
prior_mean = t.randn(N, F)
prior_mean_ = prior_mean[..., None]
A = t.randn(N, F, F)
prior_cov = A @ A.mT
prior_prec = t.inverse(prior_cov)

ap_mean = t.randn(N, F)
B = t.randn(N, F, F)
ap_cov = B@B.mT + 2*t.eye(F)

C = t.randn(N, F, F)
like_cov = C @ C.mT
like_prec = t.inverse(like_cov)

data = 1.5+t.randn(N, F)
data_ = data[..., None]
post_prec = prior_prec + like_prec
post_cov = t.inverse(post_prec)
post_mean_ = post_cov @ (prior_prec@prior_mean_ + like_prec@data_)
post_mean = post_mean_.squeeze(-1)

P = Plate(
    a = MultivariateNormal(prior_mean, prior_cov),
    d = MultivariateNormal('a', like_cov),
)

Q = Plate(
    a = MultivariateNormal(ap_mean, ap_cov),
    d = Data(),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {}
data = {'d': data}
problem = Problem(P, Q, all_platesizes, data)

moments = [('a', mean)]
known_moments = {
    ('a', mean): post_mean,
}

tp = TestProblem(problem, moments, known_moments=known_moments, moment_K=1000000)
