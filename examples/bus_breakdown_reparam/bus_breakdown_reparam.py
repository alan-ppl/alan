import torch as t
from alan import Normal, Binomial, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam
import math 
M, J, I = 3, 3, 30

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False):
    platesizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}
    all_platesizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':2*I}

    covariates = {'run_type': t.load(f'{data_dir}run_type_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load(f'{data_dir}bus_company_name_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.load(f'{data_dir}run_type_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load(f'{data_dir}bus_company_name_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat((covariates['run_type'],test_covariates['run_type']),2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],2)}
    
    if not fake_data:
        data = {'obs':t.load(f'{data_dir}delay_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        test_data = {'obs':t.load(f'{data_dir}delay_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    else:
        P = get_P(all_platesizes, all_covariates)
        all_data = {'obs': P.sample()['obs'].align_to('plate_Year', 'plate_Borough', 'plate_ID')}

        data = {'obs': all_data['obs'][:,:,:I]}

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def get_P(platesizes, covariates):
    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    P = Plate(
        log_sigma_phi_psi = Normal(0, 1),

        psi = Normal(t.zeros((run_type_dim,)), t.ones((run_type_dim,))),
        phi = Normal(t.zeros((bus_company_name_dim,)), t.ones((bus_company_name_dim,))),

        sigma_beta = Normal(0, 1),
        mu_beta = Normal(0, 1),

        plate_Year = Plate(
            beta = Normal(lambda mu_beta: mu_beta / 100, lambda sigma_beta: (sigma_beta).exp() / 100),

            sigma_alpha = Normal(0, 1),

            plate_Borough = Plate(
                alpha = Normal(lambda beta: 100 * beta, lambda sigma_alpha: (sigma_alpha).exp()),
        
                plate_ID = Plate(
                    obs = Binomial(total_count=131, logits = lambda alpha, phi, psi, run_type, bus_company_name: alpha + phi @ bus_company_name + psi @ run_type),
                )
            )
        )

    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":

        Q = Plate(
            global_latents = Group(
                log_sigma_phi_psi = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                psi = Normal(OptParam(t.zeros(run_type_dim)), OptParam(t.zeros(run_type_dim), transformation=t.exp)),
                phi = Normal(OptParam(t.zeros(bus_company_name_dim)), OptParam(t.zeros(bus_company_name_dim), transformation=t.exp)),

                sigma_beta = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                mu_beta = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            ),
            plate_Year = Plate(
                year_latents = Group(
                    beta = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                    sigma_alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                ),
                plate_Borough = Plate(
                    alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                    plate_ID = Plate(
                        obs = Data()
                    )
                )
            )
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)

    else:
        assert Q_param_type == "qem"

        Q = Plate(
            global_latents = Group(
                log_sigma_phi_psi = Normal(QEMParam(0.), QEMParam(1.)),

                psi = Normal(QEMParam(t.zeros((run_type_dim,))), QEMParam(t.ones((run_type_dim,)))),
                phi = Normal(QEMParam(t.zeros((bus_company_name_dim,))), QEMParam(t.ones((bus_company_name_dim,)))),

                sigma_beta = Normal(QEMParam(0.), QEMParam(1.)),
                mu_beta = Normal(QEMParam(0.), QEMParam(1.)),
            ),
            plate_Year = Plate(
                year_latents = Group(
                    beta = Normal(QEMParam(0.), QEMParam(1.)),

                    sigma_alpha = Normal(QEMParam(0.), QEMParam(1.)),
                ),
                plate_Borough = Plate(
                    alpha = Normal(QEMParam(0.), QEMParam(1.)),

                    plate_ID = Plate(
                        obs = Data()
                    )
                )
            )
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

    basic_runner.run('bus_breakdown_reparam',
                     methods = ['vi', 'rws', 'qem'],
                     K = 3,
                     num_runs = 1,
                     num_iters = 10,
                     lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1},
                     fake_data = False,
                     device = 'cpu')