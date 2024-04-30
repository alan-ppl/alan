import alan
import torch
import numpy as np
import stan
import arviz as az

code = """
data {
  int<lower=0> J;
  array[J] real x;
}
parameters {
  real mu;
  real log_sigma;
}
transformed parameters {
  real sigma = exp(log_sigma);
}
model {
  target += normal_lpdf(mu | 0, 1);
  target += normal_lpdf(log_sigma | 0, 1);
  target += normal_lpdf(x | mu, sigma);
}
"""

J = 1000
data = 20+10*np.random.randn(J)

data_alan = {"x": torch.tensor(data, names=('J',))}
data_stan = {"J": J, "x": data}

posterior = stan.build(code, data=data_stan)
fit = posterior.sample(num_chains=4, num_samples=1000)
print(az.summary(fit))

P = alan.Plate(
    mu = alan.Normal(0, 1),
    log_sigma = alan.Normal(0, 1),
    J = alan.Plate(
        x = alan.Normal('mu', lambda log_sigma: log_sigma.exp())
    ),
)

Q = alan.Plate(
    mu = alan.Normal(0, 20),
    log_sigma = alan.Normal(0, 10),
    J = alan.Plate(
        x = alan.Data(),
    ),
)

all_platesizes = {'J': J}
P = alan.BoundPlate(P, all_platesizes)
Q = alan.BoundPlate(Q, all_platesizes)

prob = alan.Problem(P, Q, data_alan)
sample = prob.sample(K=1000)
