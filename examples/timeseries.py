import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries, Data

P = Plate( 
    ts1_init = Normal(0., 1.),
    ts2_init = Normal(0., 1.),
    T = Plate(
        ts1 = Timeseries('ts1_init', Normal(lambda prev: 0.9*prev, 0.1)),
        ts2 = Timeseries('ts2_init', Normal(lambda ts1, prev: 0.9*ts1 + prev, 0.1)),
        a = Normal('ts2', 1.)
    ),
)

Q = Plate( 
    ts1_init = Normal(0., 1.),
    ts2_init = Normal(0., 1.),
    T = Plate(
        ts1 = Normal(0., 1.),
        ts2 = Normal(0., 1.),
        a = Data(),
    ),
)

bP = BoundPlate(P, {'T': 3})
bQ = BoundPlate(Q, {'T': 3})

data = {'a': bP.sample()['a']}

problem = Problem(bP, bQ, data)
sample = problem.sample(K=10)

elbo = sample.elbo_vi()

