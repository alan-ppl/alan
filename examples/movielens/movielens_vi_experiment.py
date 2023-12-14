import torch as t
from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample
from alan_simplified.IndexedSample import IndexedSample
import pickle

Ks = [3,10,30,100,300,1000]
lrs = [0.0001, 0.001, 0.01]
num_runs = 10
num_iters = 100

d_z = 18
M, N = 300, 5

platesizes = {'plate_1': M, 'plate_2': N}
all_platesizes = {'plate_1': M, 'plate_2': 2*N}

covariates = {'x':t.load(f'data/weights_{N}_{M}.pt')}
test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}.pt')}
all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

data = {'obs':t.load(f'data/data_y_{N}_{M}.pt')}
test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}.pt')}
all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
data['obs'] = data['obs'].rename('plate_1','plate_2')
all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')


elbos = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs))
p_lls = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs))

for num_run in range(num_runs):
    for K_idx, K in enumerate(Ks):
        for lr_idx, lr in enumerate(lrs):
            t.manual_seed(num_run)
            print(f"K: {K}, lr: {lr}, num_run: {num_run}")

            P = Plate(
                mu_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),
                psi_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),

                plate_1 = Plate(
                    z = Normal("mu_z", lambda psi_z: psi_z.exp()),

                    plate_2 = Plate(
                        obs = Bernoulli(logits = lambda z, x: z @ x),
                    )
                ),
            )

            Q = Plate(
                mu_z = Normal("mu_z_loc", "mu_z_scale"),
                psi_z = Normal("psi_z_loc", "psi_z_scale"),

                plate_1 = Plate(
                    z = Normal("z_loc", "z_scale"),

                    plate_2 = Plate(
                    )
                ),
            )

            P = BoundPlate(P, inputs = covariates)

            Q = BoundPlate(Q, inputs = covariates,
                              params = {"mu_z_loc":   t.zeros((d_z,)), 
                                        "mu_z_scale": t.ones((d_z,)),
                                        "psi_z_loc":   t.zeros((d_z,)), 
                                        "psi_z_scale": t.ones((d_z,)),
                                        "z_loc":   t.zeros((M, d_z), names=('plate_1', None)),
                                        "z_scale": t.ones((M, d_z), names=('plate_1', None))})

            prob = Problem(P, Q, platesizes, data)

            sampling_type = IndependentSample

            opt = t.optim.Adam(prob.Q.parameters(), lr=lr)

            for iter in range(num_iters+1):
                opt.zero_grad()

                sample = prob.sample(K, True, sampling_type)
                elbo = sample.elbo()

                post_idxs = sample.sample_posterior(num_samples=10)
                isample = IndexedSample(sample, post_idxs)

                ll = isample.predictive_ll(prob.P, all_platesizes, True, all_data, all_covariates)
                # print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")

                elbos[K_idx, lr_idx, iter, num_run] = elbo.item()
                p_lls[K_idx, lr_idx, iter, num_run] = ll['obs'].item()

                if iter < num_iters:
                    (-elbo).backward()
                    opt.step()

to_pickle = {'elbos': elbos, 'p_lls': p_lls, 'Ks': Ks, 'lrs': lrs, 'num_runs': num_runs, 'num_iters': num_iters}
with open('results/results.pkl', 'wb') as f:
    pickle.dump(to_pickle, f)