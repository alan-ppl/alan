import torch as t
from alan_simplified import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, MultivariateNormal
from TestProblem import TestProblem

F = 2
prior_mean = t.randn(F)
A = t.randn(F, F)
prior_cov = A @ A.mT
prior_prec = t.inverse(prior_cov)

ap_mean = t.randn(F)
B = t.randn(F, F)
ap_cov = B@B.mT + 4*t.eye(F)

C = t.randn(F, F)
like_cov = C @ C.mT
like_prec = t.inverse(like_cov)

N = 10
data = 1.5+t.randn(N, F)
post_prec = prior_prec + data.shape[0]*like_prec
post_cov = t.inverse(post_prec)
post_mean = post_cov @ (prior_prec@prior_mean + like_prec@data.sum(0))


P = Plate(
    a = MultivariateNormal(prior_mean, prior_cov),
    T = Plate(
        d = MultivariateNormal('a', like_cov),
    ),
)

Q = Plate(
    a = MultivariateNormal(ap_mean, ap_cov),
    T = Plate(
        d = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': N}
data = {'d': data.refine_names('T', None)}
problem = Problem(P, Q, all_platesizes, data)

moments = [('a', mean)]
known_moments = {
    ('a', mean): post_mean,
}

tp = TestProblem(problem, moments, moment_K=10000)
