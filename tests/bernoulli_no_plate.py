import torch as t
from alan import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, Split
from TestProblem import TestProblem

P = Plate(
    p = Beta(2, 1),
    T = Plate(
        coin = Bernoulli('p'),
    ),
)

Q = Plate(
    p = Beta(1, 1),
    T = Plate(
        coin = Data(),
    ),
)

P = BoundPlate(P)
Q = BoundPlate(Q)

all_platesizes = {'T': 10}
data = {'coin': t.cat([t.zeros(3), t.ones(7)]).refine_names('T')}

moments = [('p', mean)]
known_moments = {('p', mean): (7+2)/(2+1+10)}

tp = TestProblem(
    P, Q, all_platesizes, data, 
    moments, 
    known_moments=known_moments, 
    moment_K=10000, 
    computation_strategy=Split('T', 4),
)
