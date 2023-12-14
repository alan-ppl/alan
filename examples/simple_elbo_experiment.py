import torch as t
from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample
from alan_simplified.IndexedSample import IndexedSample

num_runs = 500
Ks = [1,10,100]

# Scalar-valued random variable model
print("Vector-valued random variable model")
elbos = t.zeros((len(Ks), num_runs))

for num_run in range(num_runs):
    # if num_run % 100 == 0: 
    #     print(f"num_run: {num_run}")
    for K_idx, K in enumerate(Ks):
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
                ),
            ),
        )

        Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

        platesizes = {'p1': 3, 'p2': 4}
        data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

        prob = Problem(P, Q, platesizes, data)

        sampling_type = IndependentSample
        sample = prob.sample(K, True, sampling_type)

        elbo = sample.elbo()

        elbos[K_idx, num_run] = elbo

for K_idx, K in enumerate(Ks):
    print(f"K: {Ks[K_idx]}, elbo: {elbos[K_idx,:].mean()}")

print()

# Vector-valued random variable model
print("Vector-valued random variable model")

elbos = t.zeros((len(Ks), num_runs))

d = 25

for num_run in range(num_runs):
    # if num_run % 100 == 0: 
    #     print(f"num_run: {num_run}")
    for K_idx, K in enumerate(Ks):
        P = Plate(
            ab = Group(
                a = Normal(t.zeros((d,)), t.ones((d,))),
                b = Normal("a", t.ones((d,))),
            ),
            c = Normal(t.ones((d,)), lambda a: a.exp()),
            p1 = Plate(
                d = Normal("a", t.ones((d,))),
                p2 = Plate(
                    e = Normal("d", t.ones((d,))),
                ),
            ),
        )

        Q = Plate(
            ab = Group(
                a = Normal("a_mean", t.ones((d,))),
                b = Normal("a", t.ones((d,))),
            ),
            c = Normal(t.zeros((d,)), lambda a: a.exp()),
            p1 = Plate(
                d = Normal("d_mean", t.ones((d,))),
                p2 = Plate(
                ),
            ),
        )

        Q = BoundPlate(Q, params={'a_mean': t.zeros((d,)), 'd_mean':t.zeros((3,d), names=('p1',None))})

        platesizes = {'p1': 3, 'p2': 4}
        data = {'e': t.randn(3, 4, d, names=('p1', 'p2', None))}

        prob = Problem(P, Q, platesizes, data)

        sampling_type = IndependentSample
        sample = prob.sample(K, True, sampling_type)

        elbo = sample.elbo()

        elbos[K_idx, num_run] = elbo

for K_idx, K in enumerate(Ks):
    print(f"K: {Ks[K_idx]}, elbo: {elbos[K_idx,:].mean()}")