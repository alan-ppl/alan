import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, mean, Split
from TestProblem import TestProblem

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
Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(4, names=('p1',))})

all_platesizes = {'p1': 4, 'p2': 4}
data = {'e': t.randn(4, 4, names=('p1', 'p2'))}

moments = [
    ('a', mean),
    ('b', mean),
    ('c', mean),
    ('d', mean),
]
tp = TestProblem(
    P, Q, all_platesizes, data, 
    moments, 
    moment_K=1000, 
    computation_strategy=Split('p1', 3)
)
