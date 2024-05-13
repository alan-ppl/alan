import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam, mean, mean2

N = 4
z_mean = 33
z_var = 0.5
obs_var = 10

N_extended = 8

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    platesizes = {'plate_1': N}
    all_platesizes = {'plate_1': N_extended}

    if not fake_data:
        data = {'obs':t.load(f'{data_dir}data_{N}.pt')}
        test_data = {'obs':t.load(f'{data_dir}test_data_{N}.pt')}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1',)}
        data['obs'] = data['obs'].rename('plate_1')
        all_data['obs'] = all_data['obs'].rename('plate_1')

    else:
        P = get_P(platesizes, {})
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('plate_1')}

        data = {'obs': all_data['obs']}

        all_latents = sample
        latents = sample 

        if return_fake_latents:
            return platesizes, all_platesizes, data, all_data, {}, {}, latents, all_latents

    return platesizes, all_platesizes, data, all_data, {}, {}

def get_P(platesizes, covariates):
    P = Plate(
        mean = Normal(z_mean, z_var),

        plate_1 = Plate(
            obs = Normal('mean', obs_var),
        ),
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":
        Q = Plate(
            mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            plate_1 = Plate(
                obs = Data()
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)#,


    else:
        assert Q_param_type == 'qem'

        Q = Plate(
            mean = Normal(QEMParam(0.), QEMParam(1.)),
            plate_1 = Plate(
                obs = Data()
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)

    prob = Problem(P, Q, data)
    prob.to(device)

    return prob

def _load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/', fake_data=False):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir, fake_data)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":

    import os, sys
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))
    import basic_runner

    basic_runner.run('synthetic_model',
                     methods=['qem'],
                     K = 10,
                     num_runs = 1,
                     num_iters = 100,
                     lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1},
                     fake_data = False,
                     device = 'cpu')
    

