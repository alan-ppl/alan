import torch as t
from alan import Normal, NegativeBinomial, Timeseries, Plate, BoundPlate, Problem, Data, QEMParam, OptParam
import math
nRs = 92
nWs = 21
nCMs = 11

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    platesizes = {'nRs':nRs,
               'nWs':int(nWs*0.8)}

    all_platesizes = {'nRs':nRs,
                    'nWs':nWs}

    covariates = {'ActiveCMs_NPIs': t.load(f'{data_dir}ActiveCMs_NPIs.pt').rename('nRs', 'nWs', None), 
                  'ActiveCMs_wearing': t.load(f'{data_dir}ActiveCMs_wearing.pt').rename('nRs', 'nWs'), 
                  'ActiveCMs_mobility': t.load(f'{data_dir}ActiveCMs_mobility.pt').rename('nRs', 'nWs')}
    all_covariates = {'ActiveCMs_NPIs': t.load(f'{data_dir}ActiveCMs_NPIs_all.pt').rename('nRs', 'nWs', None).to(device), 
                      'ActiveCMs_wearing': t.load(f'{data_dir}ActiveCMs_wearing_all.pt').rename('nRs', 'nWs').to(device),
                      'ActiveCMs_mobility': t.load(f'{data_dir}ActiveCMs_mobility_all.pt').rename('nRs', 'nWs').to(device),}
    
    if not fake_data:
        data = {'obs': t.load(f'{data_dir}obs.pt').rename('nRs', 'nWs' ),}
        all_data = {'obs': t.load(f'{data_dir}obs_all.pt').rename('nRs', 'nWs' ).to(device),}

    else:
        P = get_P(all_platesizes, all_covariates)
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('nRs', 'nWs' )}

        data = {'obs': all_data['obs'][:,:int(nWs*0.8)]}

        all_latents = sample
        latents = sample 

        if return_fake_latents:
            return platesizes, all_platesizes, data, all_data, covariates, all_covariates, latents, all_latents

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def get_P(platesizes, covariates):
    cm_prior_scale=1
    wearing_mean=0
    wearing_sigma=0.4
    mobility_mean=1.704
    mobility_sigma=0.44
    R_prior_mean_mean=1.07
    R_prior_mean_scale=0.2
    R_noise_scale=0.4
    
    Expected_Log_Rs = lambda RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, prev: RegionR - \
                        CM_alpha@ActiveCMs_NPIs - Wearing_alpha*ActiveCMs_wearing - Mobility_alpha*ActiveCMs_mobility + prev

    P = Plate(
        #Effect of NPI
        CM_alpha = Normal(0, cm_prior_scale, sample_shape=[nCMs-2]),
        #Effect of mask wearing
        Wearing_alpha = Normal(wearing_mean, wearing_sigma),
        #Effect of mobility restrictions
        Mobility_alpha = Normal(mobility_mean, mobility_sigma),
        #R for each region
        RegionR = Normal(R_prior_mean_mean, R_prior_mean_scale + R_noise_scale),

        nRs = Plate(
            #Initial number of infected in each region
            InitialSize_log = Normal(0, 1),
            nWs = Plate(
            
                log_infected = Timeseries('InitialSize_log', Normal(Expected_Log_Rs, 0.1)),
                #Rate parameter for Poisson
                Psi = Normal(math.log(1000), 1),
                #Observations
                obs = NegativeBinomial(total_count=lambda Psi: t.exp(Psi), logits=lambda Psi, log_infected: t.exp(Psi)/(t.exp(Psi) + t.exp(log_infected) + 1e-4) ),
            ),
        ),  
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":

        Q = Plate(
            CM_alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp), sample_shape=[nCMs-2]),
            Wearing_alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            Mobility_alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            RegionR = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            #Expected_Log_Rs = lambda RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility: RegionR - CM_alpha*ActiveCMs_NPIs - Wearing_alpha*ActiveCMs_wearing - Mobility_alpha*ActiveCMs_mobility,
            
            nRs = Plate(
                InitialSize_log = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                nWs = Plate(
                    #log_infected = Timeseries('InitialSize_log', Normal(lambda prev, RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility: prev + RegionR - CM_alpha@ActiveCMs_NPIs - Wearing_alpha*ActiveCMs_wearing - Mobility_alpha*ActiveCMs_mobility, 0.1)),
                    log_infected = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                    Psi = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                    obs = Data()
                ),
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)

    else:
        assert Q_param_type == "qem"

        Q = Plate(
            CM_alpha = Normal(QEMParam(t.zeros((nCMs-2,))), QEMParam(t.ones((nCMs-2,)))),
            Wearing_alpha = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            Mobility_alpha = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            RegionR = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            nRs = Plate(
                InitialSize_log = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                nWs = Plate(
                    log_infected = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                    Psi = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                    obs = Data()
                ),
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

    basic_runner.run('covid',
                     methods = ['vi', 'rws', 'qem'],
                     K = 3,
                     num_runs = 1,
                     num_iters = 10,
                     lrs = {'vi': 0.1, 'rws': 0.1, 'qem': 0.1},
                     fake_data = False,
                     device = 'cuda')