#Bayesian logistic regression multi-class classification on MNIST
import torch as t
from alan import Normal, Categorical, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

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

# Model
P = Plate(
    w = Normal(t.zeros((28**2,10)), 1),
    plate1 = Plate(
        y = Categorical(logits= lambda w, x: x.flatten() @ w),
    ),
)

Q = Plate(
    w = Normal('w_mean', 1),
    plate1 = Plate(
        y = Data()
    ),
)

all_platesizes = {'plate1': trainset.data.shape[0]}

data = {'y': trainset.targets.rename('plate1')}
inputs = {'x': trainset.data.rename('plate1', ...)}

test_data = {'y': testset.targets.rename('plate1')}
test_inputs = {'x': testset.data.rename('plate1', ...)}


P_bound = BoundPlate(P, all_platesizes, inputs=inputs)
Q_bound = BoundPlate(Q, all_platesizes, extra_opt_params={'w_mean': t.zeros((28**2,10))})

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



