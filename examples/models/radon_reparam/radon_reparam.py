## Radon model in 919 houses and 85 counties from Gelman et al. (2006)
import torch as t
from alan import Normal, Bernoulli, HalfNormal, Plate, BoundPlate, Problem, Data, mean, OptParam, QEMParam, checkpoint, no_checkpoint, Split, Group

import numpy as np
from pathlib import Path
import os
import math

t.manual_seed(123)


def load_data_covariates(device, run, data_dir="data", fake_data=False, return_fake_latents=False):
    #Load tensors and rename
    log_radon = t.load(os.path.join(data_dir, "log_radon.pt"))
    basement = t.load(os.path.join(data_dir, "basement.pt"))
    log_uranium = t.load(os.path.join(data_dir, "log_u.pt"))

    # shuffle along Zips dimension
    # perm = t.randperm(log_radon.shape[examples/chimpanzees/results/K5_10_15_lr_0.001-1

    # platesizes = {'States': log_radon.shape[0], 'Counties': log_radon.shape[1], 'Zips': int(log_radon.shape[2] * 0.9)}
    # all_platesizes = {'States': log_radon.shape[0], 'Counties': log_radon.shape[1], 'Zips': log_radon.shape[2]}

    # train_log_radon = {'obs': log_radon[:, :, :platesizes['Zips']].rename('States', 'Counties', 'Zips')}
    # all_log_radon = {'obs': log_radon.float().rename('States', 'Counties', 'Zips')}

    # train_inputs = {'basement': basement[:, :, :platesizes['Zips']].rename('States', 'Counties', 'Zips'),
    #                 'log_uranium': log_uranium[:, :, :platesizes['Zips']].rename('States', 'Counties', 'Zips')}
    
    # all_inputs = {'basement': basement.rename('States', 'Counties', 'Zips'),
    #                 'log_uranium': log_uranium.rename('States', 'Counties', 'Zips')}

    platesizes = {'States': log_radon.shape[0], 'Zips': log_radon.shape[1] // 2}
    all_platesizes = {'States': log_radon.shape[0], 'Zips': log_radon.shape[1]}

    train_inputs = {'basement': basement[:, :platesizes['Zips']].rename('States', 'Zips'),
                    'log_uranium': log_uranium[:, :platesizes['Zips']].rename('States', 'Zips')}
    
    all_inputs = {'basement': basement.rename('States', 'Zips'),
                    'log_uranium': log_uranium.rename('States', 'Zips')}
    
    if not fake_data:
        train_log_radon = {'obs': log_radon[:, :platesizes['Zips']].rename('States', 'Zips')}
        all_log_radon = {'obs': log_radon.float().rename('States', 'Zips')}

    else:
        P = get_P(all_platesizes, all_inputs)
        sample = P.sample()
        all_log_radon = {'obs': sample.pop('obs').align_to('States', 'Zips')}

        train_log_radon = {'obs': all_log_radon['obs'][:, :platesizes['Zips']].rename('States', 'Zips')}

        all_latents = sample
        latents = sample

        if return_fake_latents:
            return platesizes, all_platesizes, train_log_radon, all_log_radon, train_inputs, all_inputs, latents, all_latents
    

    return platesizes, all_platesizes,  train_log_radon, all_log_radon, train_inputs, all_inputs

def get_P(platesizes, covariates):
    
    P = Plate( 
        global_mean = Normal(0., 1.),
        global_log_sigma = Normal(0., 1.),
        States = Plate(
            State_mean = Normal(lambda global_mean: global_mean/1000, lambda global_log_sigma: global_log_sigma.exp()/1000),
            State_log_sigma = Normal(0., 1.),
            Beta_u = Normal(0., 1.),
            Beta_basement = Normal(0., 1.),

            Zips = Plate( 
                obs = Normal(lambda State_mean, basement, log_uranium, Beta_basement, Beta_u: 1000*State_mean + basement*Beta_basement + log_uranium * Beta_u, lambda State_log_sigma: State_log_sigma.exp()),
            ),
        ),
    )
    
    P = BoundPlate(P, platesizes, inputs=covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P_bound_plate = get_P(platesizes, covariates)

    if Q_param_type == "opt": 
        Q_plate = Plate(
            global_latents = Group(
                global_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                global_log_sigma = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            ),
            States = Plate(
                    State_mean = Normal(OptParam(0.), OptParam(-math.log(1000), transformation=t.exp)),
                    State_log_sigma = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                    Beta_u = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                    Beta_basement = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                Zips = Plate(
                    obs = Data(),
                ),
            ),
        ) 
        
    elif Q_param_type == "qem":
        Q_plate = Plate(
            global_latents = Group(
                global_mean = Normal(QEMParam(0.), QEMParam(1.)),
                global_log_sigma = Normal(QEMParam(0.), QEMParam(1.)),
            ),
            States = Plate(
                State_mean = Normal(QEMParam(0.), QEMParam(1/1000)),
                State_log_sigma = Normal(QEMParam(0.), QEMParam(1.)),
                Beta_u = Normal(QEMParam(0.), QEMParam(1.)),
                Beta_basement = Normal(QEMParam(0.), QEMParam(1.)),
            Zips = Plate(
                obs = Data(),
            ),
        ),
    )
     
    Q_bound_plate = BoundPlate(Q_plate, platesizes, inputs=covariates)

    prob = Problem(P_bound_plate, Q_bound_plate, data)
    prob.to(device)

    return prob



def _load_and_generate_problem(device, Q_param_type, run=0, data_dir='data', fake_data=False):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir, fake_data)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    import os, sys
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))
    import basic_runner

    basic_runner.run('radon_reparam',
                     K = 3,
                     methods=['vi', 'rws', 'qem'],
                     num_runs = 1,
                     num_iters = 5,
                     lrs = {'vi': 0.1, 'rws': 0.3, 'qem': 0.1},
                     fake_data = False,
                     device = 'cpu')
    