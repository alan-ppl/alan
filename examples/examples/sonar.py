# Sonar dataset from https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

import numpy as np

t.manual_seed(123)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
computation_strategy = checkpoint

## Load and preprocess data
targets = np.genfromtxt("data/sonar.all-data", delimiter=",", usecols=60, converters={60: lambda x: 1 if x == b"R" else 0})
inputs = np.genfromtxt("data/sonar.all-data", delimiter=",", usecols=range(60))

train_y = t.tensor(targets[:150]).float()
test_y = t.tensor(targets[150:]).float()

train_x = t.tensor(inputs[:150]).float()
test_x = t.tensor(inputs[150:]).float()

#Append 1s for bias
train_x = t.cat([train_x, t.ones(train_x.shape[0], 1)], dim=1).float()
test_x = t.cat([test_x, t.ones(test_x.shape[0], 1)], dim=1).float()



## Baysian logistic regression model
N_feat = train_x.shape[1]
P_plate = Plate( 
    mu = Normal(t.zeros((N_feat,)), t.ones((N_feat,))),
    plate1 = Plate(
        y = Bernoulli(logits=lambda mu, x: mu @ x)
    ),   
)

Q_plate = Plate(
    mu = Normal(OptParam(t.zeros((N_feat,))), t.ones((N_feat,))),
    plate1 = Plate(
        y = Data()
    ),   
)

all_platesizes = {'plate1': train_x.shape[0]}

data = {'y': train_y.rename('plate1')}
inputs = {'x': train_x.rename('plate1', ...)}

P_bound_plate = BoundPlate(P_plate, all_platesizes, inputs=inputs)
Q_bound_plate = BoundPlate(Q_plate, all_platesizes)

prob = Problem(P_bound_plate, Q_bound_plate, data)
prob.to(device)

opt = t.optim.Adam(prob.parameters(), lr=1e-4)
        
K = 20
print("K={}".format(K))
for i in range(3000):
    opt.zero_grad()
    sample = prob.sample(K=K)
    elbo = sample.elbo_vi(computation_strategy=computation_strategy)
    elbo.backward()
    opt.step()

    if 0 == i%200:
        print(elbo.item())