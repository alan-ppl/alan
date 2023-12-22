import torch as t
from TestProblem import TestProblem
from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, Data

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

all_platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, all_platesizes, data)
tp = TestProblem(prob)
