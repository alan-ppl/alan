import torch as t
from alan_simplified import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data

P = Plate(
    p = Beta(2, 1),
    p1 = Plate(
        d = Bernoulli(p),
    ),
)

Q = Plate(
    p = Beta(1, 1),
    p1 = Plate(
        d = Data(p),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'p1': 10}
data = {'e': t.cat([t.zeros(3), t.ones(7)])}
problem = Problem(P, Q, all_platesizes, data)

known_means = {p: (7+1)/(2+1+10)}

tp = TestProblem(problem, known_means=known_means)
