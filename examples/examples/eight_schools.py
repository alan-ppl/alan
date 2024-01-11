###
# From Bayesian Data Analysis, section 5.5 (Gelman et al. 2013):

# A study was performed for the Educational Testing Service to analyze the effects of special coaching programs for SAT-V 
# (Scholastic Aptitude Test-Verbal) in each of eight high schools. The outcome variable in each study was the score on a special 
# administration of the SAT-V, a standardized multiple choice test administered by the Educational Testing Service and used to 
# help colleges make admissions decisions; the scores can vary between 200 and 800, with mean about 500 and standard deviation 
# about 100. The SAT examinations are designed to be resistant to short-term efforts directed specifically toward improving 
# performance on the test; instead they are designed to reflect knowledge acquired and abilities developed over many years of 
# education. Nevertheless, each of the eight schools in this study considered its short-term coaching program to be very 
# successful at increasing SAT scores. Also, there was no prior reason to believe that any of the eight programs was more 
# effective than any other or that some were more similar in effect to each other than to any other.
###       
import torch as t
from alan import Normal, HalfCauchy, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

import numpy as np
from pathlib import Path

from posteriordb import PosteriorDatabase
import os


def load_data_covariates(device, run, data_dir="data"):
    pdb_path = os.path.join(os.getcwd(), "posteriordb/posterior_database")
    my_pdb = PosteriorDatabase(pdb_path)

    posterior = my_pdb.posterior("eight_schools-eight_schools_centered")

    data = posterior.data.values()

    #Treatment effects
    train_y = t.tensor(data["y"][:6]).rename('J')
    all_y = t.tensor(data["y"]).rename('J')
    #Standard errors
    train_sigma = t.tensor(data["sigma"][:6]).rename('J')
    all_sigma = t.tensor(data["sigma"]).rename('J')
    
    platesizes = {'J': 6}
    all_platesizes = {'J': 8}

    train_y = {'obs': train_y}
    all_y = {'obs': all_y}
    
    train_sigma = {'sigma': train_sigma}
    all_sigma = {'sigma': all_sigma}

    return platesizes, all_platesizes,  train_y, all_y, train_sigma, all_sigma

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    
    P_plate = Plate( 
        tau = Normal(0, 1),
        log_mu_scale = Normal(0, 1),
        J = Plate(
            mu = Normal(0, lambda log_mu_scale: log_mu_scale.exp()),
            theta = Normal('mu', lambda tau: tau.exp()),
            obs = Normal('theta', 'sigma'),
        ),   
    )

    

    if Q_param_type == "opt": 
        Q_plate = Plate( 
            tau = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            log_mu_scale = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            J = Plate(
                mu = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                theta = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
                obs = Data(),
            ),   
        )
    elif Q_param_type == "qem":
        Q_plate = Plate( 
            tau = Normal(QEMParam(0.), QEMParam(1.)),
            log_mu_scale = Normal(QEMParam(0.), QEMParam(1.)),
            J = Plate(
                mu = Normal(QEMParam(0.), QEMParam(1.)),
                theta = Normal(QEMParam(0.), QEMParam(1.)),
                obs = Data(),
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
    Path("plots/eight_schools").mkdir(parents=True, exist_ok=True)
    DO_PLOT   = True
    DO_PREDLL = True
    NUM_ITERS = 100
    NUM_RUNS  = 1

    K = 10

    vi_lr = 0.01
    rws_lr = 0.01
    qem_lr = 0.01

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
        plt.title(f'Eight Schools (K={K})')
        plt.tight_layout()
        plt.savefig('plots/eight_schools/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'Eight Schools (K={K})')
            plt.tight_layout()
            plt.savefig('plots/eight_schools/quick_predlls.png')