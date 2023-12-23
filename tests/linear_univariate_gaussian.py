import torch as t
from alan_simplified import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, IndependentSample, Normal
from TestProblem import TestProblem

prior_mean = 2
prior_scale = 2
prior_prec = 1/prior_scale**2

like_scale = 3
like_prec = 1/like_scale**2

mult=2.5

data = 1.5+t.randn(10)

post_prec = prior_prec + data.shape[0]*like_prec*mult**2
post_mean = (prior_prec*prior_mean + like_prec*mult**2*(data.sum()/mult)) / post_prec


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

all_platesizes = {'T': 10}
data = {'d': data.refine_names('T')}
problem = Problem(P, Q, all_platesizes, data)

moments = [('a', mean), ('a', mean2)]
known_moments = {
    ('a', mean): post_mean,
    ('a', mean): post_mean**2 + 1/post_prec,
}

tp = TestProblem(problem, moments, known_moments=known_moments, moment_K=10000)

