import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam, mean, mean2

import math
d_z = 18
M, N = 300, 5

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    platesizes = {'plate_1': M, 'plate_2': N}
    all_platesizes = {'plate_1': M, 'plate_2': 2*N}

    covariates = {'x':t.load(f'{data_dir}weights_{N}_{M}.pt')}
    test_covariates = {'x':t.load(f'{data_dir}test_weights_{N}_{M}.pt')}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

    if not fake_data:
        data = {'obs':t.load(f'{data_dir}data_y_{N}_{M}.pt')}
        test_data = {'obs':t.load(f'{data_dir}test_data_y_{N}_{M}.pt')}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
        data['obs'] = data['obs'].rename('plate_1','plate_2')
        all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')

    else:
        P = get_P(all_platesizes, all_covariates)
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('plate_1', 'plate_2')}

        data = {'obs': all_data['obs'][:,:N]}

        all_latents = sample
        latents = sample 

        if return_fake_latents:
            return platesizes, all_platesizes, data, all_data, covariates, all_covariates, latents, all_latents

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def get_P(platesizes, covariates):
    logits = lambda z, x: (100*z) @ x
    P = Plate(
        mu_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),
        psi_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),

        plate_1 = Plate(
            z = Normal(lambda mu_z: 1/100 * mu_z, lambda psi_z:1/100 * psi_z.exp()),

            plate_2 = Plate(
                obs = Bernoulli(logits = logits),
            )
        ),
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":
        Q = Plate(
            mu_z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),
            psi_z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),

            plate_1 = Plate(
                z = Normal(OptParam(t.zeros((d_z,))), OptParam(-math.log(100) * t.zeros((d_z,)), transformation=t.exp)),

                plate_2 = Plate(
                    obs = Data()
                )
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)#,


    else:
        assert Q_param_type == 'qem'

        Q = Plate(
            mu_z = Normal(QEMParam(t.zeros((d_z,))), QEMParam(t.ones((d_z,)))),
            psi_z = Normal(QEMParam(t.zeros((d_z,))), QEMParam(t.ones((d_z,)))),

            plate_1 = Plate(
                z = Normal(QEMParam(t.zeros((d_z,))), QEMParam(1/100 * t.ones((d_z,)))),

                plate_2 = Plate(
                    obs = Data()
                )
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
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    import basic_runner

    basic_runner.run('movielens_reparam',
                     methods=['qem'],
                     K = 10,
                     num_runs = 1,
                     num_iters = 100,
                     lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1},
                     fake_data = False,
                     device = 'cpu')
    
    # platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates('cpu', fake_data=False)
    
    # P = get_P(platesizes, covariates)
    
    # prior_samples = P.sample(1000)
    # # for name, sample in prior_samples.items():
    # #     print(name, sample.mean(0), sample.std(0))
    # for name, sample in prior_samples.items():
    #     print(name, sample.mean('N'), sample.std('N'))
        
    # problem, all_data, all_covariates, all_platesizes = _load_and_generate_problem('cpu', 'qem', run=0, data_dir='data/', fake_data=False)
    
    # psi_means = []
    # psi_vars = []
    # z_means = []
    # z_vars = []
    
    # #posterior means
    # print('posterior means')
    # for _ in range(100):
    #     posterior_samples = problem.sample(10)
    #     moments = [('psi_z', mean), ('psi_z', mean2), ('z', mean), ('z', mean2)]
        
    #     posterior_means = posterior_samples.moments(moments)
    #     psi_means.append(posterior_means[0])
    #     psi_vars.append(posterior_means[1] - posterior_means[0]**2)
    #     z_means.append(posterior_means[2].rename(None))
    #     z_vars.append(posterior_means[3].rename(None) - posterior_means[2].rename(None)**2)
    
    # print('psi_z')
    # print(t.stack(psi_means).mean(0))
    # print(t.stack(psi_vars).mean(0))
    # print('z')
    # print(t.stack(z_means).mean(0))
    # print(t.stack(z_vars).mean(0))
    
    

