import torch as t
from alan import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, Normal, PermutationSampler, CategoricalSampler
from TestProblem import TestProblem

prior_mean = t.randn(2)
prior_scale = t.tensor([1., 2.])
prior_var = prior_scale**2
prior_prec = 1/prior_var

z_scale = t.tensor([1.3, 1.6])
d_scale = t.tensor([2., 3.])

like_var = z_scale**2 + d_scale**2
like_prec = 1/like_var

N = 10
data = 1.5+t.randn(N, 2).refine_names('T', None)
post_prec = prior_prec + data.shape[0]*like_prec
post_mean = (prior_prec*prior_mean + like_prec*data.sum('T')) / post_prec

P = Plate(
    a = Normal(prior_mean, prior_scale),
    T = Plate(
        z = Normal('a', z_scale),
        d = Normal('z', d_scale),
    ),
)

Q = Plate(
    a = Normal(t.zeros(2), 4),
    T = Plate(
        z = Normal(lambda a: 0.5*a, 6),
        d = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': N}
data = {'d': data}
problem = Problem(P, Q, all_platesizes, data)

moments = [('a', mean), ('a', mean2), ('z', mean), ('z', mean2)]
known_moments = {
    ('a', mean): post_mean,
    ('a', mean2): post_mean**2 + 1/post_prec,
}



tp = TestProblem(
    problem, 
    moments, 
    known_moments=known_moments, 
    moment_K=1000, 
)

