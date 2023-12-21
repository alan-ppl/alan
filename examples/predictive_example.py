import torch as t
from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample, Data

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

P = BoundPlate(P)
Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, platesizes, data)

# Get some initial samples (with K dims)
sampling_type = IndependentSample
sample = prob.sample(5, True, sampling_type)

# Get importance samples (with N dims)
importance_sample = sample.importance_sample(num_samples=10)

# Get predictive (extended) samples (with N dims)
extended_platesizes = {'p1': 5, 'p2': 6}
predictive_samples = importance_sample.extend(extended_platesizes, True, None)
print(predictive_samples.dump())

# Sample fake extended data (really the first (3,4) of this should be the same as the original data)
test_data = {'e': t.randn(5, 6, names=('p1', 'p2'))}

# Get predictive log likelihood of the extended data
pll = predictive_samples.predictive_ll(test_data)
print(pll)

# breakpoint()

