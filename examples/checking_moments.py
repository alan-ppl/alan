import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, mean, var, mean2
from functorch.dim import Dim
import math
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


P_simple = Plate(mu = Normal(0, 1), 
                        p1 = Plate(obs = Normal("mu", 1)))
        
Q_simple = Plate(mu = Normal("mu_mean", 1),
                 p1 = Plate(obs = Data()))

platesizes_simple = {'p1': 3}
data = {'obs': data.refine_names('p1')}

P = BoundPlate(P_simple, platesizes_simple)
Q = BoundPlate(Q_simple, platesizes_simple, extra_opt_params={'mu_mean': t.zeros(())})

prob = Problem(P, Q, data)

# Get some initial samples (with K dims)
sample = prob.sample(2000, True)

# for K in [1,3,10,30,100]:
#     print(prob.sample(K, True, sampling_type).elbo())
# # Obtain K indices from posterior
# post_idxs = sample.sample_posterior(num_samples=10)

# def mean(x):
#     sample = x
#     dim = x.dims[0]
    
#     w = 1/dim.size
#     return (w * sample).sum(dim)

# def second_moment(x):
#     return mean(t.square(x))

# def square(x):
#     return x**2

# def var(x):
#     return mean(square(x)) - square(mean(x))


# # print(sample.sample)
# K_dim = sample.sample['mu'].dims[0]
# # print(data['obs'])
# p1 = Dim('p1', 3)
# lps = Normal(sample.sample['mu'], 1).log_prob(data['obs'].rename(None)[p1], scope={})


# print(t.logsumexp(lps.order(K_dim).sum(p1), dim=0) - math.log(K_dim.size))


# print(sample.elbo())


marginals = sample.marginals()
moments = marginals.moments([('mu', mean), ('mu', var), ('mu', mean2)])
print(moments)

#Getting moments from posterior samples:

importance_samples = sample.importance_sample(N=4000).dump()

print(f'mean: {importance_samples["mu"].mean("N")}')
print(f'variance: {(importance_samples["mu"]**2).mean("N") - importance_samples["mu"].mean("N")**2}')
print(f'second moment: {(importance_samples["mu"]**2).mean("N")}')

### using posterior
posterior_samples = t.distributions.Normal(posterior_mean, posterior_var**(1/2)).sample((10000,))

print(f'posterior mean: {posterior_samples.mean()}')
print(f'posterior variance: {(posterior_samples**2).mean() - posterior_samples.mean()**2}')
print(f'posterior second moment: {(posterior_samples**2).mean()}')

