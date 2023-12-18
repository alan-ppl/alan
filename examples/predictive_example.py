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

# Obtain K indices from posterior
# post_idxs = sample.sample_posterior(num_samples=10)


# Sample some fake test data 
# (technically we should ensure that test_data includes the original data)
extended_platesizes = {'p1': 5, 'p2': 6}
test_data = {'e': t.randn(5, 6, names=('p1', 'p2'))}

# Compute predictive samples and predictive log likelihood
predictive_samples = sample.predictive_sample(extended_platesizes, True, num_samples=10)
print(predictive_samples)

pll = sample.predictive_ll(extended_platesizes, True, test_data, num_samples=10)
print(pll)

# breakpoint()

