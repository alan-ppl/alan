#Probabilistic PCA on MNIST data
import torch as t
from alan import Normal, MultivariateNormal, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

t.manual_seed(123)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
computation_strategy = Split('plate1', 32)

#Get MNIST from torch
import torchvision

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True)

testset = torchvision.datasets.MNIST(root='./data', train=False)

#Cast to float 
trainset.data = trainset.data.float()
testset.data = testset.data.float()

# Centering the data
trainset.data = trainset.data - trainset.data.mean()
testset.data = testset.data - testset.data.mean()

# Flatten the data
trainset.data = trainset.data.flatten(1)
testset.data = testset.data.flatten(1)


K = 16
lamb = 1
# Model
P = Plate(
    w = Normal(t.zeros((28**2,K)), 1),
    plate1 = Plate(
        x = MultivariateNormal(t.zeros((28**2,)), lambda w: w @ w.t() + lamb**2 * t.eye(28**2)),
    ),
)

Q = Plate(
    w = Normal('w_mean', 1),
    plate1 = Plate(
        x = Data(),
    ),
)

all_platesizes = {'plate1': trainset.data.shape[0]}

data = {'x': trainset.data.rename('plate1', ...)}

test_data = {'x': testset.data.rename('plate1', ...)}


P_bound = BoundPlate(P, all_platesizes)
Q_bound = BoundPlate(Q, all_platesizes, extra_opt_params={'w_mean': t.zeros((28**2,K))})

prob = Problem(P_bound, Q_bound, data)
prob.to(device)

opt = t.optim.Adam(prob.parameters(), lr=1e-3)
        
K = 10
print("K={}".format(K))
for i in range(1000):
    opt.zero_grad()
    sample = prob.sample(K=K)
    elbo = sample.elbo_vi(computation_strategy=computation_strategy)
    elbo.backward()
    opt.step()

    if 0 == i%200:
        print(elbo.item())



