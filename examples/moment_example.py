import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, mean, var, mean2

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

platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

P = BoundPlate(P, platesizes)
Q = BoundPlate(Q, platesizes, extra_opt_params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

prob = Problem(P, Q, data)

# Get some initial samples (with K dims)
sample = prob.sample(3, True)

print("ELBO")
for K in [1,3,10,30,100]:
    print(prob.sample(K, True).elbo_nograd())
print()

print("Moments from marginals:")
marginals = sample.marginals()
marginal_moments = marginals.moments([('d', mean), ('d', mean2), ('c', mean2), ('c', var)])

print(marginal_moments)

print("Moments from approximate posterior samples:")
moments = sample.moments((('d', mean), ('d', mean2), ('c', mean2)))
print(moments)

# # below doesn't work because var is not a raw moment
# # print(sample.moments(('d', var)))

# print()

# #Getting moments from posterior samples:
# print("Moments from importance samples:")
# importance_samples = sample.importance_sample(N=1000)

# posterior_moments = importance_samples.moments((('d', mean), ('d', mean2), ('c', mean2)))
# print(posterior_moments)

# print()

# print("Moments directly from dumped importance samples:")
# importance_samples_flat = importance_samples.dump()

# print(importance_samples_flat['d'].mean('N'))
# print((importance_samples_flat['d']**2).mean('N'))

# print((importance_samples_flat['c']**2).mean('N'))