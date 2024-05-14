import torch as t
from alan import Normal, Binomial, Bernoulli, ContinuousBernoulli, Uniform, Beta, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam
import math 

t.manual_seed(123)

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    M, J, I, Returns = 6, 12, 200, 5
    I_extended = 300

    platesizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I, 'plate_Replicate': Returns}
    all_platesizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I_extended, 'plate_Replicate': Returns}

    # if splitting on Replicates not on Ids:
    # platesizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I_extended, 'plate_Replicate': 3}
    # all_platesizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I_extended, 'plate_Replicate': 5}

    covariates = {'weather': t.load(f'{data_dir}weather_train_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load(f'{data_dir}quality_train_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    test_covariates = {'weather': t.load(f'{data_dir}weather_test_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load(f'{data_dir}quality_test_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    all_covariates = {'weather': t.cat([covariates['weather'],test_covariates['weather']],-1),
        'quality': t.cat([covariates['quality'],test_covariates['quality']],-1)}
    
    if not fake_data:
        data = {'obs':t.load(f'{data_dir}birds_train_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}
        test_data = {'obs':t.load(f'{data_dir}birds_test_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}

        all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}

        # if splitting on Replicates not on Ids:
        # all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

        data['obs'] = data['obs'].float()
        all_data['obs'] = all_data['obs'].float()

    else:
        P = get_P(all_platesizes, all_covariates)
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('plate_Years', 'plate_Birds', 'plate_Ids', 'plate_Replicate')}

        data = {'obs': all_data['obs'][:,:,:I,:]}

        all_latents = sample
        latents = sample 
        latents['z'] = latents['z'][:,:,:I]

        if return_fake_latents:
            return platesizes, all_platesizes, data, all_data, covariates, all_covariates, latents, all_latents

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def get_P(platesizes, covariates):
    P = Plate(
        # how common is any bird?
        bird_mean_mean = Normal(0., 1.), 
        bird_mean_log_var = Normal(0., 1.),

        # alpha = effect of quality on bird - how easy it is to see
        alpha_mean = Normal(0., 1.),
        alpha_log_var = Normal(0., 1.),

        # beta = effect of weather on bird - how common it is hot weather (-> "temperature") 
        beta_mean = Normal(0., 1.),
        beta_log_var = Normal(0., 1.),

        plate_Birds = Plate(
            bird_mean = Normal('bird_mean_mean', lambda bird_mean_log_var: bird_mean_log_var.exp()), # how common is this bird?

            alpha = Normal('alpha_mean', lambda alpha_log_var: alpha_log_var.exp()), # how easy is this bird to see?

            beta = Normal('beta_mean', lambda beta_log_var: beta_log_var.exp()), # how much does weather affect this bird?

            plate_Years = Plate(
                bird_year_mean = Normal(lambda bird_mean: 1/1000 * bird_mean, 1/1000 * 1.), # how common is this bird this year?

                plate_Ids = Plate(
                    
                    z = Bernoulli(logits=lambda weather, bird_year_mean, beta: 1000*bird_year_mean*weather*beta), # was this bird actually present?

                    plate_Replicate = Plate(
                        obs = Bernoulli(logits=lambda alpha, quality, z: alpha * quality * z + (1-z)*(-10)) # did we actually see this bird?
                    )
                ),
            )
        )
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":
        Q = Plate(
            global_latents = Group(
                bird_mean_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
                bird_mean_log_var = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),

                alpha_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
                alpha_log_var = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),

                beta_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
                beta_log_var = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
            ),

            plate_Birds = Plate(
                bird_latents = Group(
                    bird_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how common is this bird?

                    alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how easy is this bird to see?

                    beta = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how much does weather affect this bird?
                ),
                plate_Years = Plate(
                    bird_year_mean = Normal(OptParam(0.), OptParam(-math.log(1000), transformation=t.exp),), # how common is this bird this year?

                    plate_Ids = Plate(
                        
                        z = Bernoulli(logits=lambda weather, bird_year_mean, beta: 1000*bird_year_mean*weather*beta), # was this bird actually present?

                        plate_Replicate = Plate(
                            obs = Data()
                        )
                    ),
                )
            )
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)
    else:
        assert Q_param_type == 'qem'

        Q = Plate(
            global_latents = Group(
                bird_mean_mean = Normal(QEMParam(0.), QEMParam(1.),),
                bird_mean_log_var = Normal(QEMParam(0.), QEMParam(1.),),

                alpha_mean = Normal(QEMParam(0.), QEMParam(1.),),
                alpha_log_var = Normal(QEMParam(0.), QEMParam(1.),),

                beta_mean = Normal(QEMParam(0.), QEMParam(1.),),
                beta_log_var = Normal(QEMParam(0.), QEMParam(1.),),
            ),
            plate_Birds = Plate(
                bird_latents = Group(
                    bird_mean = Normal(QEMParam(0.), QEMParam(1.),), # how common is this bird?

                    alpha = Normal(QEMParam(0.), QEMParam(1.),), # how easy is this bird to see?

                    beta = Normal(QEMParam(0.), QEMParam(1.),), # how much does weather affect this bird?
                ),
                plate_Years = Plate(
                    bird_year_mean = Normal(QEMParam(0.), QEMParam(1/1000),), # how common is this bird this year?

                    plate_Ids = Plate(
                        
                        z = Bernoulli(logits=lambda weather, bird_year_mean, beta: 1000*bird_year_mean*weather*beta), # was this bird actually present?

                        plate_Replicate = Plate(
                            obs = Data()

                        )
                    ),
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

    # if splitting on Replicates not on Ids:
    # all_covariates = covariates
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    import os, sys
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))
    import basic_runner

    basic_runner.run('occupancy_reparam',
                     methods = ['rws', 'qem'],
                     K = 3,
                     num_runs = 1,
                     num_iters = 10,
                     lrs = {'rws': 0.1, 'qem': 0.1},
                     reparam = False,
                     fake_data = False,
                     device = 'cpu')
    