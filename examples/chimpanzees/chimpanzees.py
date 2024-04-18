import torch as t
from alan import Normal, Bernoulli, HalfCauchy, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    # num_actors, num_actors_extended = 6, 7
    # num_blocks, num_blocks_extended = 4, 6
    num_actors, num_blocks = 7, 6
    num_repeats, num_repeats_extended = 10, 12

    platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks, 'plate_repeats': num_repeats}
    # all_platesizes = {'plate_actors': num_actors_extended, 'plate_blocks': num_blocks_extended}
    all_platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks, 'plate_repeats': num_repeats_extended}

    # platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks}
    # all_platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks}

    covariates      = {'condition':   t.load(f'{data_dir}condition_train.pt'),
                       'prosoc_left': t.load(f'{data_dir}prosoc_left_train.pt')}
    test_covariates = {'condition':   t.load(f'{data_dir}condition_test.pt'),
                       'prosoc_left': t.load(f'{data_dir}prosoc_left_test.pt')}
    all_covariates  = {'condition':   t.cat([covariates['condition'],
                                             test_covariates['condition']],-1).rename('plate_actors','plate_blocks','plate_repeats'),
                       'prosoc_left': t.cat([covariates['prosoc_left'],
                                             test_covariates['prosoc_left']],-1).rename('plate_actors','plate_blocks','plate_repeats')}
    
    covariates['condition'] = covariates['condition'].rename('plate_actors','plate_blocks','plate_repeats')
    covariates['prosoc_left'] = covariates['prosoc_left'].rename('plate_actors','plate_blocks','plate_repeats')

    if not fake_data:
        data = {'obs':t.load(f'{data_dir}data_train.pt')}
        test_data = {'obs':t.load(f'{data_dir}data_test.pt')}
        all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_actors','plate_blocks','plate_repeats')}
        
        data['obs'] = data['obs'].rename('plate_actors','plate_blocks','plate_repeats')
    
    else:
        P = get_P(all_platesizes, all_covariates)
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('plate_actors','plate_blocks','plate_repeats')}

        data = {'obs': all_data['obs'][:,:,:num_repeats]}

        all_latents = sample
        latents = sample 

        if return_fake_latents:
            return platesizes, all_platesizes, data, all_data, covariates, all_covariates, latents, all_latents

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def get_P(platesizes, covariates):
    P = Plate(
        sigma_block = HalfCauchy(1.),
        sigma_actor = HalfCauchy(1.),

        beta_PC = Normal(0., 10.),
        beta_P = Normal(0., 10.),

        alpha = Normal(0., 10.),

        plate_actors = Plate(
            alpha_actor = Normal(0., 'sigma_actor'),

            plate_blocks = Plate(
                alpha_block = Normal(0., 'sigma_block'),

                plate_repeats = Plate(
                    obs = Bernoulli(logits=lambda alpha, alpha_block, alpha_actor, beta_PC, beta_P, condition, prosoc_left: alpha + alpha_actor + alpha_block + (beta_P + beta_PC*condition)*prosoc_left),
                )
            )
        ),
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":
        Q = Plate(
            global_latents = Group(
                sigma_block = HalfCauchy(OptParam(1.)),
                sigma_actor = HalfCauchy(OptParam(1.)),

                # sigma_block = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                # sigma_actor = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                beta_PC = Normal(OptParam(0.), OptParam(t.tensor(10.).log(), transformation=t.exp)),
                beta_P = Normal(OptParam(0.), OptParam(t.tensor(10.).log(), transformation=t.exp)),

                alpha = Normal(OptParam(0.), OptParam(t.tensor(10.).log(), transformation=t.exp)),
            ),
            plate_actors = Plate(
                alpha_actor = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                plate_blocks = Plate(
                    alpha_block = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                    plate_repeats = Plate(
                        obs = Data()
                    )
                )
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)
        
    else:
        assert Q_param_type == 'qem'

        Q = Plate(
            global_latents = Group(
                sigma_block = HalfCauchy(OptParam(1.)),
                sigma_actor = HalfCauchy(OptParam(1.)),

                beta_PC = Normal(QEMParam(0.), QEMParam(t.tensor(10.))),
                beta_P = Normal(QEMParam(0.), QEMParam(t.tensor(10.))),

                alpha = Normal(QEMParam(0.), QEMParam(t.tensor(10.))),
            ),
            plate_actors = Plate(
                alpha_actor = Normal(QEMParam(0.), QEMParam(1.)),

                plate_blocks = Plate(
                    alpha_block = Normal(QEMParam(0.), QEMParam(1.)),

                    plate_repeats = Plate(
                        obs = Data()
                    )
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

    basic_runner.run('chimpanzees',
                     methods = ['vi', 'rws', 'qem'],
                     K = 3,
                     num_runs = 1,
                     num_iters = 10,
                     lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1},
                     fake_data = False,
                     device = 'cpu')