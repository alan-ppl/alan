import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, mean, Split, OptParam, QEMParam
from TestProblem import TestProblem

P = Plate( ab = Group(
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
        a = Normal(QEMParam(0.), QEMParam(1.)),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal(OptParam(0.), "d_scale"),
        p2 = Plate(
            e = Data()
        ),
    ),
)


all_platesizes = {'p1': 4, 'p2': 4}


extra_opt_params = {'d_scale': t.ones(4, names=('p1',))}
P = BoundPlate(P, all_platesizes)
Q = BoundPlate(Q, all_platesizes, extra_opt_params=extra_opt_params)

data = {'e': t.randn(4, 4, names=('p1', 'p2'))}
moments = [
    ('a', mean),
    ('b', mean),
    ('c', mean),
    ('d', mean),
]
tp = TestProblem(
    P, Q, data,
    moments, 
    moment_K=1000, 
    computation_strategy=Split('p1', 3)
)

problem = tp.problem

sample = problem.sample(K=10)

sample.update_qem_params(0.1)

