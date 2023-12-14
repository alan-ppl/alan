import torch as t
from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample, Data
from alan_simplified.IndexedSample import IndexedSample

t.manual_seed(0)

P = Plate(
    ab = Group(
        a = Normal(0, 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
            e = Normal("d", 1.),
        ),
    ),
)

Q = Plate(
    ab = Group(
        a = Normal("a_mean", 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("d_mean", 1),
        p2 = Plate(
            e = Data()
        ),
    ),
)

Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, platesizes, data)

# Get some initial samples (with K dims)
sampling_type = IndependentSample
sample = prob.sample(5, True, sampling_type)

# Obtain K indices from posterior
post_idxs = sample.sample_posterior(num_samples=10)

# Create posterior samples explicitly using sample and post_idxs
isample = IndexedSample(sample, post_idxs)


def mean(x):
    sample = x
    dim = x.dims[-1]

    w = 1/dim.size
    return (w * sample).sum(dim)

def second_moment(x):
    return mean(t.square(x))

def square(x):
    return x**2

def var(x):
    return mean(square(x)) - square(mean(x))

moments = sample.moments({'d': [mean, var], 'c': [second_moment]}, post_idxs, isample)
print(moments)