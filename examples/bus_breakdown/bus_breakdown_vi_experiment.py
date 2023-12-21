import torch as t
import torchopt
from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample, Data
import pickle
import time
from bus_breakdown import load_data_covariates, generate_problem

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'

Ks = [3,10,30]#,100]
lrs = [0.0001, 0.001, 0.01]
num_runs = 10
num_iters = 1000

d_z = 18
M, N = 300, 5

platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device)

elbos = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs)).to(device)
p_lls = t.zeros((len(Ks), len(lrs), num_iters+1, num_runs)).to(device)

with open("job_status.txt", "w") as f:
    f.write(f"Starting job.\n")

for num_run in range(num_runs):
    for K_idx, K in enumerate(Ks):
        K_start_time = time.time()
        for lr_idx, lr in enumerate(lrs):
            t.manual_seed(num_run)
            print(f"K: {K}, lr: {lr}, num_run: {num_run}")

            prob = generate_problem(device, platesizes, data, covariates)

            sampling_type = IndependentSample

            # opt = t.optim.Adam(prob.Q.parameters(), lr=lr)
            opt = torchopt.Adam(prob.Q.parameters(), lr=lr)

            for i in range(num_iters+1):
                opt.zero_grad()

                sample = prob.sample(K, True, sampling_type)
                elbo = sample.elbo()

                importance_sample = sample.importance_sample(num_samples=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, False, all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)

                if i % 50 == 0 or True:
                    print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")

                elbos[K_idx, lr_idx, i, num_run] = elbo.item()
                p_lls[K_idx, lr_idx, i, num_run] = ll['obs'].item()

                if i < num_iters:
                    (-elbo).backward()
                    opt.step()

        with open("job_status.txt", "a") as f:
            f.write(f"num_run: {num_run} K: {K} done in {time.time()-K_start_time}s.\n")

to_pickle = {'elbos': elbos.cpu(), 'p_lls': p_lls.cpu(), 'Ks': Ks, 'lrs': lrs, 'num_runs': num_runs, 'num_iters': num_iters}

print()

for K_idx, K in enumerate(Ks):
    for lr_idx, lr in enumerate(lrs):
        print(f"K: {K}, lr: {lr}")
        print(f"elbo: {elbos[K_idx, lr_idx, 0,:].mean():.3f}")
        print(f"p_ll: {p_lls[K_idx, lr_idx, 0,:].mean():.3f}")
        print()

# breakpoint()
with open('results/results.pkl', 'wb') as f:
    pickle.dump(to_pickle, f)