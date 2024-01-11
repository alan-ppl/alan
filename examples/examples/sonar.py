# Sonar dataset from https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split

import numpy as np
from pathlib import Path
  
def load_data_covariates(device, run, data_dir="data"):
    ## Load and preprocess data
    targets = np.genfromtxt("data/sonar.all-data", delimiter=",", usecols=60, converters={60: lambda x: 1 if x == b"R" else 0})
    inputs = np.genfromtxt("data/sonar.all-data", delimiter=",", usecols=range(60))

    train_y = t.tensor(targets[:150]).float()
    all_y = t.tensor(targets).float()

    train_x = t.tensor(inputs[:150]).float()
    all_x = t.tensor(inputs).float()

    #Append 1s for bias
    train_x = t.cat([train_x, t.ones(train_x.shape[0], 1)], dim=1).float()
    all_x = t.cat([all_x, t.ones(all_x.shape[0], 1)], dim=1).float()
    
    train_y = {'obs': train_y.rename('plate1')}
    all_y = {'obs': all_y.rename('plate1')}
    train_x = {'x': train_x.rename('plate1', ...)}
    all_x = {'x': all_x.rename('plate1', ...)}
    
    platesizes = {'plate1': train_x['x'].shape[0]}
    all_platesizes = {'plate1': all_x['x'].shape[0]}

    return platesizes, all_platesizes,  train_y, all_y, train_x, all_x

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    
    N_feat = covariates['x'].shape[1]
    P_plate = Plate( 
        log_mu_scale = Normal(0.,1.),
        plate1 = Plate(
            mu = Normal(0., lambda log_mu_scale: log_mu_scale.exp(), sample_shape = t.Size([N_feat])),
            obs = Bernoulli(logits=lambda mu, x: mu @ x),
        ),   
    )


    if Q_param_type == "opt": 
        Q_plate = Plate(
            log_mu_scale = Normal(OptParam(0.), OptParam(1., transformation=t.exp)),
            plate1 = Plate(
                mu = Normal(OptParam(0.), OptParam(1.,  transformation=t.exp), sample_shape = t.Size([N_feat])),
                obs = Data(),
            ),   
        ) 
    elif Q_param_type == "qem":
        Q_plate = Plate(
            log_mu_scale = Normal(QEMParam(0.), QEMParam(1.)),
            plate1 = Plate(
                mu = Normal(QEMParam(t.zeros(N_feat,)), QEMParam(t.ones(N_feat,))),
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
    Path("plots/sonar").mkdir(parents=True, exist_ok=True)
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
        plt.ylim(-1000,0)
        plt.title(f'Sonar (K={K})')
        plt.tight_layout()
        plt.savefig('plots/sonar/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.ylim(-200,0)
            plt.title(f'Sonar (K={K})')
            plt.tight_layout()
            plt.savefig('plots/sonar/quick_predlls.png')