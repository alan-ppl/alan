import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam, Split

d_z = 18
M, N = 943, 841

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    platesizes = {'plate_1': M, 'plate_2': N}
    all_platesizes = {'plate_1': M, 'plate_2': 2*N}

    covariates = {'x':t.load(f'{data_dir}weights_{N}_{M}_{run}.pt')}
    test_covariates = {'x':t.load(f'{data_dir}test_weights_{N}_{M}_{run}.pt')}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

    if not fake_data:
        data = {'obs':t.load(f'{data_dir}data_y_{N}_{M}_{run}.pt')}
        test_data = {'obs':t.load(f'{data_dir}test_data_y_{N}_{M}_{run}.pt')}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
        data['obs'] = data['obs'].rename('plate_1','plate_2')
        all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')

    else:
        P = get_P(all_platesizes, all_covariates)
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('plate_1', 'plate_2')}

        data = {'obs': all_data['obs'][:,:N]}

        if return_fake_latents:
            return platesizes, all_platesizes, data, all_data, covariates, all_covariates, sample

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def get_P(platesizes, covariates):
    P = Plate(
        # mu_z_global_mean = Normal(0., 1.),
        # mu_z_global_log_scale = Normal(0., 1.),
        # mu_z = Normal("mu_z_global_mean", 
        #               lambda mu_z_global_log_scale: mu_z_global_log_scale.exp(), 
        #               sample_shape = t.Size([d_z]),
        # ),

        # psi_z_global_mean = Normal(0., 1.),
        # psi_z_global_log_scale = Normal(0., 1.),
        # psi_z = Normal("psi_z_global_mean", 
        #                lambda psi_z_global_log_scale: psi_z_global_log_scale.exp(), 
        #                sample_shape = t.Size([d_z]),
        # ),

        mu_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),
        psi_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),

        plate_1 = Plate(
            z = Normal("mu_z", lambda psi_z: psi_z.exp()),

            plate_2 = Plate(
                obs = Bernoulli(logits = lambda z, x: z @ x),
            )
        ),
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":
        Q = Plate(
            # mu_z_global_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            # mu_z_global_log_scale = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            # mu_z = Normal("mu_z_global_mean", 
            #               lambda mu_z_global_log_scale: mu_z_global_log_scale.exp(), 
            #               sample_shape = t.Size([d_z]),
            # ),

            # psi_z_global_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            # psi_z_global_log_scale = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            # psi_z = Normal("psi_z_global_mean", 
            #               lambda psi_z_global_log_scale: psi_z_global_log_scale.exp(), 
            #               sample_shape = t.Size([d_z]),
            # ),

            mu_z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),
            psi_z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),

            plate_1 = Plate(
                # z = Normal("z_mean", lambda z_log_scale: z_log_scale.exp()),
                z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),

                plate_2 = Plate(
                    obs = Data()
                )
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)#,
                        # extra_opt_params = {"z_mean":   t.zeros((M, d_z), names=('plate_1', None)),
                        #                     "z_log_scale": t.zeros((M, d_z), names=('plate_1', None))})

    else:
        assert Q_param_type == 'qem'

        Q = Plate(
            # mu_z_global_mean = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            # mu_z_global_log_scale = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            # mu_z = Normal("mu_z_global_mean", 
            #               lambda mu_z_global_log_scale: mu_z_global_log_scale.exp(), 
            #               sample_shape = t.Size([d_z]),
            # ),

            # psi_z_global_mean = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            # psi_z_global_log_scale = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            # psi_z = Normal("psi_z_global_mean", 
            #               lambda psi_z_global_log_scale: psi_z_global_log_scale.exp(), 
            #               sample_shape = t.Size([d_z]),
            # ),

            mu_z = Normal(QEMParam(t.zeros((d_z,))), QEMParam(t.ones((d_z,)))),
            psi_z = Normal(QEMParam(t.zeros((d_z,))), QEMParam(t.ones((d_z,)))),

            plate_1 = Plate(
                z = Normal(QEMParam(t.zeros((d_z,))), QEMParam(t.ones((d_z,)))),

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

    basic_runner.run('movielens_FULL',
                     K = 10,
                     num_runs = 1,
                     num_iters = 10,
                     lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1},
                     fake_data = False,
                     device = 'cuda',
                     split = Split('plate_1', 50))
