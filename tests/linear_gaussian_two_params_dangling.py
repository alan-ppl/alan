import torch as t
from alan_simplified import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, Normal
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
    b = Normal('a', 1.3),
    T = Plate(
        d = Normal(lambda a: mult*a, like_scale),
    ),
)

Q = Plate(
    a = Normal(1, 4),
    b = Normal(lambda a: 1.2*a, 1.2),
    T = Plate(
        d = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': N}
data = {'d': data.refine_names('T')}
problem = Problem(P, Q, all_platesizes, data)

moments = [('a', mean), ('a', mean2), ('b', mean), ('b', mean2)]
known_moments = {
    ('a', mean): post_mean,
    ('a', mean2): post_mean**2 + 1/post_prec,
    ('b', mean): post_mean,
    ('b', mean2): post_mean**2 + 1/post_prec + 1.3**2,
}



tp = TestProblem(problem, moments, known_moments=known_moments, known_elbo=known_elbo, moment_K=1000, elbo_K=1000)

