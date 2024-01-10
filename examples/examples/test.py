import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Data, checkpoint, OptParam, QEMParam

computation_strategy = checkpoint

P = Plate(
    mu = Normal(t.zeros((2,)), t.ones((2,)), sample_shape = t.Size([2])), 
    p1 = Plate(
        obs = Normal("mu", t.ones((2,)))
    )
)
    
Q = Plate(
    mu = Normal(QEMParam(t.zeros((2,))), QEMParam(t.ones((2,))),
    p1 = Plate(
        obs = Data()
    )
)


platesizes = {'p1': 3}


P = BoundPlate(P, platesizes)
Q = BoundPlate(Q, platesizes)

P_sample = P.sample()
data = {'obs': P_sample['obs']}

prob = Problem(P, Q, data)

sample = prob.sample(K=10)
sample.update_qem_params(0.01)