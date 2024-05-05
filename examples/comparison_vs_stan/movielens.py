import alan
import torch
import numpy as np
import stan
import arviz as az
torch.manual_seed(0)

# D is the number of user/film features
# movies is the number of films
# users is the number of users
# ratings is binary, describing how each user rated each film.
# film_features is a bunch of fixed, known features describing each film.
# z is a latent feature vector for each user.

D = 2 #Number of features
movies = 10  #Number of movies
users  = 20  #Number of users
all_platesizes = {'movies':movies, 'users':users}
movie_features_stan = torch.randn(movies, D).numpy()

movie_features_alan = torch.tensor(movie_features_stan, dtype=torch.float32)
movie_features_alan = movie_features_alan[:, None, :].expand(-1, users, -1)
movie_features_alan = movie_features_alan.refine_names('movies', 'users', None)

P = alan.Plate(
    group = alan.Group(
        mu_z      = alan.Normal(torch.zeros((D,)), torch.ones((D,))),
        log_psi_z = alan.Normal(torch.zeros((D,)), torch.ones((D,))),
    ),

    users = alan.Plate(
        z = alan.Normal("mu_z", lambda log_psi_z: log_psi_z.exp()),

        movies = alan.Plate(
            ratings = alan.Bernoulli(logits = lambda z, movie_features: torch.dot(z, movie_features))
        )
    ),
)
P = alan.BoundPlate(P, all_platesizes, inputs={'movie_features': movie_features_alan})
prior_sample = P.sample()
ratings = prior_sample["ratings"].align_to('users', 'movies')
data_alan = {'ratings': ratings}


Q = alan.Plate(
    mu_z = alan.Normal(
         alan.QEMParam(torch.zeros((D,))), 
         alan.QEMParam(torch.ones((D,))), 
    ),
    log_psi_z = alan.Normal(
         alan.QEMParam(torch.zeros((D,))),
         alan.QEMParam(torch.ones((D,))), 
    ),

    users = alan.Plate(
        z = alan.Normal(
            alan.QEMParam(torch.zeros((D,))), 
            alan.QEMParam(torch.ones((D,))), 
        ),

        movies = alan.Plate(
            ratings = alan.Data()
        )
    ),
)
Q = alan.BoundPlate(Q, all_platesizes, inputs={'movie_features': movie_features_alan})
prob = alan.Problem(P, Q, data_alan)

for i in range(200):
    prob.sample(K=3).update_qem_params(0.1)

def std(qem_means, varname):
    return torch.sqrt(qem_means[f'{varname}_mean2'] - qem_means[f'{varname}_mean']**2)

#Stan matmul is * (elementwise product is .*)

code = """
data {
  int<lower=0> D;
  int<lower=0> movies;
  int<lower=0> users;
  array[users,movies] int<lower=0,upper=1> ratings;
  matrix[movies,D] movie_features;
}
parameters {
  vector[D] mu_z;
  vector[D] log_psi_z;
  matrix[users,D] z; 
}
transformed parameters {
  vector[D] psi_z = exp(log_psi_z);
  matrix[users,movies] logits = z * movie_features';
}
model {
  target += normal_lpdf(mu_z | 0, 1);
  target += normal_lpdf(log_psi_z | 0, 1);
  for (user in 1:users)
    for (d in 1:D)
      target += normal_lpdf(z[user, d] | mu_z[d], psi_z[d]);
  for (user in 1:users)
    for (movie in 1:movies)
      target += bernoulli_logit_lpmf(ratings[user, movie] | logits[user, movie]);
}
"""

data_stan = {
    'D': D, 
    'movies': movies, 
    'users': users, 
    'movie_features': movie_features_stan,
    'ratings': ratings.numpy().astype(int)
}

posterior = stan.build(code, data=data_stan)
fit = posterior.sample(num_chains=4, num_samples=4000, num_warmup=4000)
print(az.summary(fit))

print(Q.qem_means())
print("Standard deviation of mu_z")
print(std(Q.qem_means(), 'mu_z'))
print("Standard deviation of log_psi_z")
print(std(Q.qem_means(), 'log_psi_z'))

