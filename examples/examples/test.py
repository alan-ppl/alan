import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Problem, Data, checkpoint, OptParam, QEMParam

computation_strategy = checkpoint

P = Plate(
    mu = Normal(0, 1, sample_shape = t.Size([2])), 
    p1 = Plate(
        theta = Normal('mu', 1, sample_shape = t.Size([2])),
        obs = Bernoulli(logits = lambda theta, x: theta @ x)
    )
)
    
Q = Plate(
    mu = Normal(QEMParam(t.zeros((2,))), QEMParam(t.ones((2,)))),
    p1 = Plate(
        theta = Normal('mu', 1),
        obs = Data()
    )
)


platesizes = {'p1': 3}

inputs = {'x': t.randn((3, 2)).rename('p1', ...)}
P = BoundPlate(P, platesizes, inputs=inputs)
Q = BoundPlate(Q, platesizes)

P_sample = P.sample()
data = {'obs': t.tensor([1., 1., 0.], names=('p1',))}

prob = Problem(P, Q, data)

sample = prob.sample(K=10)
sample.update_qem_params(0.01)