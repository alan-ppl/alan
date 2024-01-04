import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data

t.manual_seed(127)

data = t.tensor([0,1,2])
extended_data = t.tensor([0,1,2,3])
#Doing true pred_ll first because changing num_samples changes the result of predictive_ll if we do this after for some reason
extended_platesizes = {'p1': 4}
extended_data = {'obs': extended_data.refine_names('p1')} 


posterior_mean = (3+1)**(-1) * (3*t.tensor([0.0,1.0,2.0]).mean() + 0)
posterior_var = (3+1)**(-1)
#By hand pred_ll
pred_dist = t.distributions.Normal(posterior_mean, (1 + posterior_var)**(1/2))
true_pred_lik = pred_dist.log_prob(t.tensor([3.0])).sum()


P = Plate(mu = Normal(0, 1), 
          p1 = Plate(obs = Normal("mu", 1)))
        
Q = Plate(mu = Normal("mu_mean", 1),
          p1 = Plate(obs = Data()))

platesizes = {'p1': 3}
data = {'obs': data.refine_names('p1')}

P = BoundPlate(P, platesizes)
Q = BoundPlate(Q, platesizes, extra_opt_params={'mu_mean': t.zeros(())})

prob = Problem(P, Q, data)

Ks = [1,3,10,30,100,300]
Ns = [1,10,100,1000,10000,100000,1000000,10000000]
num_runs = 2

results = t.zeros((len(Ks), len(Ns), num_runs))

for i, K in enumerate(Ks):
    for j, num_samples in enumerate(Ns):
        for k in range(num_runs):
            sample = prob.sample(K, True)
            importance_sample = sample.importance_sample(num_samples)
            predictive_samples = importance_sample.extend(extended_platesizes, None)
            ll = predictive_samples.predictive_ll(extended_data)

            print(f"K={K}, N={num_samples}, run {k}: {ll['obs']}")
            results[i,j,k] = ll['obs']

results = results.detach().numpy()

import matplotlib.pyplot as plt

plt.figure()
for i in range(len(Ks)):
    plt.plot(Ns, results[i,:,:].mean(-1), label = 'K = ' + str(Ks[i]))

plt.plot(Ns, true_pred_lik.repeat(len(Ns)), label = 'True pred_ll', linestyle = 'dashed')
plt.legend()
plt.xlabel('Number of predictive samples')
plt.ylabel('Predictive log likelihood')
plt.xscale('log')
plt.title('Simple model predictive log likelihood')

plt.savefig('pred_ll_quick_test.png')
# plt.show()