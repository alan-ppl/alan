## Radon model in 919 houses and 85 counties from Gelman et al. (2006)
import torch as t
from alan import Normal, Bernoulli, HalfNormal, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

import numpy as np
from pathlib import Path
import os
from posteriordb import PosteriorDatabase

t.manual_seed(123)


def load_data_covariates(device, run, data_dir="data"):
    #Load tensors and rename
    log_radon = t.load(os.path.join(data_dir, "log_radon.pt"))
    basement = t.load(os.path.join(data_dir, "basement.pt"))
    log_uranium = t.load(os.path.join(data_dir, "log_u.pt"))

    platesizes = {'States': log_radon.shape[0], 'Counties': log_radon.shape[1], 'Zips': int(log_radon.shape[2] * 0.8)}
    all_platesizes = {'States': log_radon.shape[0], 'Counties': log_radon.shape[1], 'Zips': log_radon.shape[2]}

    train_log_radon = {'obs': log_radon[:, :, :platesizes['Zips']].rename('States', 'Counties', 'Zips')}
    all_log_radon = {'obs': log_radon.float().rename('States', 'Counties', 'Zips')}

    train_inputs = {'basement': basement[:, :, :platesizes['Zips']].rename('States', 'Counties', 'Zips'),
                    'log_uranium': log_uranium[:, :, :platesizes['Zips']].rename('States', 'Counties', 'Zips')}
    
    all_inputs = {'basement': basement.rename('States', 'Counties', 'Zips'),
                    'log_uranium': log_uranium.rename('States', 'Counties', 'Zips')}

    return platesizes, all_platesizes,  train_log_radon, all_log_radon, train_inputs, all_inputs

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    
    P_plate = Plate( 
        global_mean = Normal(0., 1.),
        global_log_sigma = Normal(0., 1.),
        States = Plate(
            State_mean = Normal('global_mean', lambda global_log_sigma: global_log_sigma.exp()),
            State_log_sigma = Normal(0., 1.),
            Counties = Plate(
                County_mean = Normal('State_mean', lambda State_log_sigma: State_log_sigma.exp()),
                County_log_sigma = Normal(0., 1.),
                Beta_u = Normal(0., 1.),
                Beta_basement = Normal(0., 1.),
                Zips = Plate( 
                    obs = Normal(lambda County_mean, basement, log_uranium, Beta_basement, Beta_u: County_mean + basement*Beta_basement + log_uranium * Beta_u, lambda County_log_sigma: County_log_sigma.exp()),
                ),
            ),
        ),
    )



    if Q_param_type == "opt": 
        Q_plate = Plate(
            global_mean = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            global_log_sigma = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            States = Plate(
                State_mean = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                State_log_sigma = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                Counties = Plate(
                    County_mean = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                    County_log_sigma = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                    Beta_u = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                    Beta_basement = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                    Zips = Plate(
                        obs = Data(),
                    ),
                ),
            ),  
        )
    elif Q_param_type == "qem":
        Q_plate = Plate(
            global_mean = Normal(QEMParam(0.), QEMParam(1.)),
            global_log_sigma = Normal(QEMParam(0.), QEMParam(1.)),
            States = Plate(
                State_mean = Normal(QEMParam(0.), QEMParam(1.)),
                State_log_sigma = Normal(QEMParam(0.), QEMParam(1.)),
                Counties = Plate(
                    County_mean = Normal(QEMParam(0.), QEMParam(1.)),
                    County_log_sigma = Normal(QEMParam(0.), QEMParam(1.)),
                    Beta_u = Normal(QEMParam(0.), QEMParam(1.)),
                    Beta_basement = Normal(QEMParam(0.), QEMParam(1.)),
                    Zips = Plate(
                        obs = Data(),
                    ),
                ),
            ),  
        )
     
    P_bound_plate = BoundPlate(P_plate, platesizes, inputs=covariates)
    Q_bound_plate = BoundPlate(Q_plate, platesizes)

    prob = Problem(P_bound_plate, Q_bound_plate, data)
    prob.to(device)

    return prob



def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    Path("plots/radon").mkdir(parents=True, exist_ok=True)
    DO_PLOT   = True
    DO_PREDLL = True
    NUM_ITERS = 100
    NUM_RUNS  = 1

    K = 10

    vi_lr = 0.05
    rws_lr = 0.001
    qem_lr = 0.1

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device='cpu'

    elbos = {'vi' : t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'rws': t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'qem': t.zeros((NUM_RUNS, NUM_ITERS)).to(device)}
    
    lls   = {'vi' : t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'rws': t.zeros((NUM_RUNS, NUM_ITERS)).to(device),
             'qem': t.zeros((NUM_RUNS, NUM_ITERS)).to(device)}

    print(f"Device: {device}")

    for num_run in range(NUM_RUNS):
        print(f"Run {num_run}")
        print()
        print(f"VI")
        t.manual_seed(num_run)
        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        opt = t.optim.Adam(prob.Q.parameters(), lr=vi_lr)


        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_vi()
            (-elbo).backward()
            opt.step()
            elbos['vi'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['vi'][num_run, i] = ll['obs']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            

        print()
        print(f"RWS")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        opt = t.optim.Adam(prob.Q.parameters(), lr=rws_lr)
        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_rws()
            elbos['rws'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['rws'][num_run, i] = ll['obs']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()
            
        print()
        print(f"QEM")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'qem')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        for i in range(NUM_ITERS):
            sample = prob.sample(K, True)
            elbo = sample.elbo_nograd()
            elbos['qem'][num_run, i] = elbo

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['qem'][num_run, i] = ll['obs']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            sample.update_qem_params(qem_lr)

    if DO_PLOT:
        for key in elbos.keys():
            elbos[key] = elbos[key].cpu()
            lls[key] = lls[key].cpu()

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t.arange(NUM_ITERS), elbos['vi'].mean(0), label=f'VI lr={vi_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['rws'].mean(0), label=f'RWS lr={rws_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['qem'].mean(0), label=f'QEM lr={qem_lr}')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title(f'Radon (K={K})')
        plt.tight_layout()
        plt.savefig('plots/radon/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'Radon (K={K})')
            plt.tight_layout()
            plt.savefig('plots/radon/quick_predlls.png')