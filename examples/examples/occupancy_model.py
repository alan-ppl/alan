# Multi-species occupancy model adapted from https://mc-stan.org/users/documentation/case-studies/dorazio-royle-occupancy.html
import torch as t
from alan import Normal, Bernoulli, Beta, Binomial, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

import numpy as np
from pathlib import Path
  
def load_data_covariates(device, run, data_dir="data"):
    ## Load and preprocess data
    data = np.genfromtxt("data/butterflyData.txt", delimiter=",", skip_header=1)[:,1:]
    print(data.shape)
    platesizes = {'species':22, 'sites':16}
    all_platesizes = {'species':28, 'sites':20}
    train_y = t.tensor(data[:22,:16]).float()
    all_y = t.tensor(data).float()
  
    train_y = {'y': train_y.rename('species', 'sites')}
    all_y = {'y': all_y.rename('species', 'sites')}

    return platesizes, all_platesizes,  train_y, all_y, {}, {}

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    
    P_plate = Plate( 
        alpha = Normal(0.,2.5),
        beta = Normal(0.,2.5),
        log_sigma = Normal(0.,1),
        species = Plate(
            u = Normal(0, lambda log_sigma: log_sigma.exp()),
            v = Normal(0, lambda log_sigma: log_sigma.exp()),
            sites = Plate(
                y = Binomial(total_count=18, logits=lambda v,beta: (v+beta)),
            ),
        ),   
    )


    if Q_param_type == "opt": 
        Q_plate = Plate(
            alpha = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            beta = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            log_sigma = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            species = Plate(
                u = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                v = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                sites = Plate(
                    y = Data()
                ),
            ),
        )
    elif Q_param_type == "qem":
        Q_plate = Plate(
            alpha = Normal(QEMParam(0.), QEMParam(1.)),
            beta = Normal(QEMParam(0.), QEMParam(1.)),
            log_sigma = Normal(QEMParam(0.), QEMParam(1.)),
            species = Plate(
                u = Normal(QEMParam(0.), QEMParam(1.)),
                v = Normal(QEMParam(0.), QEMParam(1.)),
                sites = Plate(
                    y = Data()
                ),
            ),
        )
            
    P_bound_plate = BoundPlate(P_plate, platesizes)
    Q_bound_plate = BoundPlate(Q_plate, platesizes)

    prob = Problem(P_bound_plate, Q_bound_plate, data)
    prob.to(device)

    return prob



def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    Path("plots/occupancy").mkdir(parents=True, exist_ok=True)
    DO_PLOT   = True
    DO_PREDLL = True
    NUM_ITERS = 20
    NUM_RUNS  = 1

    K = 10

    vi_lr = 0.01
    rws_lr = 0.01
    qem_lr = 0.3

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
            elbos['vi'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['vi'][num_run, i] = ll['y']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['y']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()

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

            sample = prob.sample(K, False)
            elbo = sample.elbo_rws()
            elbos['rws'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['rws'][num_run, i] = ll['y']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['y']:.3f}")
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
            sample = prob.sample(K, False)
            elbo = sample.elbo_nograd()
            elbos['qem'][num_run, i] = elbo

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['qem'][num_run, i] = ll['y']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['y']:.3f}")
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
        plt.ylim(-1000,0)
        plt.title(f'Occupancy (K={K})')
        plt.tight_layout()
        plt.savefig('plots/occupancy/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'Occupancy (K={K})')
            plt.tight_layout()
            plt.savefig('plots/occupancy/quick_predlls.png')