## Radon model in 919 houses and 85 counties from Gelman et al. (2006)
import torch as t
from alan import Normal, Bernoulli, HalfNormal, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

import numpy as np
from pathlib import Path
import os
from posteriordb import PosteriorDatabase

t.manual_seed(123)


def load_data_covariates(device, run, data_dir="data"):
    pdb_path = os.path.join(os.getcwd(), "posteriordb/posterior_database")
    my_pdb = PosteriorDatabase(pdb_path)

    posterior = my_pdb.posterior("radon_mn-radon_pooled")

    data = posterior.data.values()


    #Number of Houses
    Houses = data["N"]
    #floor measurement
    floor_measure = t.tensor(data["floor_measure"])
    #log radon measurements
    log_radon = t.tensor(data["log_radon"])
    
    train_floor_measure = {'floor_measure': floor_measure[:500].rename('Houses')}
    train_log_radon = {'obs': log_radon[:500].rename('Houses')}
    
    all_floor_measure = {'floor_measure': floor_measure.rename('Houses')}
    all_log_radon = {'obs': log_radon.rename('Houses')}
    
    platesizes = {'Houses': 500}
    all_platesizes = {'Houses': Houses}

    

    return platesizes, all_platesizes,  train_log_radon, all_log_radon, train_floor_measure, all_floor_measure

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    
    P_plate = Plate( 
        alpha = Normal(0, 10),
        sigma_y = HalfNormal(1),
        beta = Normal(0, 10),
        Houses = Plate(
            obs = Normal(lambda alpha, floor_measure, beta: alpha + floor_measure * beta , 'sigma_y'),         
        ),
    )



    if Q_param_type == "opt": 
        Q_plate = Plate( 
            alpha = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            sigma_y = HalfNormal(OptParam(1., transformation=t.exp)),
            beta = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            Houses = Plate(
                obs = Data(),         
            ),
        )
    elif Q_param_type == "qem":
        #Can't use QEM with this model
        # Q_plate = Plate( 
        #     alpha = Normal(QEMParam(0.), QEMParam(10.)),
        #     sigma_y = HalfNormal(QEMParam(1.)),
        #     beta = Normal(QEMParam(0.), QEMParam(10.)),
        #     Houses = Plate(
        #         log_radon = Data(),         
        #     ),
        # )
        None
     
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

    vi_lr = 0.1
    rws_lr = 0.1
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
            elbos['vi'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['vi'][num_run, i] = ll['obs']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
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


    if DO_PLOT:
        for key in elbos.keys():
            elbos[key] = elbos[key].cpu()
            lls[key] = lls[key].cpu()

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t.arange(NUM_ITERS), elbos['vi'].mean(0), label=f'VI lr={vi_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['rws'].mean(0), label=f'RWS lr={rws_lr}')
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
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'Radon (K={K})')
            plt.tight_layout()
            plt.savefig('plots/radon/quick_predlls.png')