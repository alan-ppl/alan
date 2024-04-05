import torch as t
from alan import Normal, NegativeBinomial, Timeseries, Plate, BoundPlate, Problem, Data, QEMParam, OptParam, Group
import math
nRs = 92
nDs = 137
nCMs = 11

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False, return_fake_latents=False):
    platesizes = {'nRs':nRs,
               'nDs':int(nDs*0.8)}

    all_platesizes = {'nRs':nRs,
                    'nDs':nDs}

    covariates = {'ActiveCMs_NPIs': t.load(f'{data_dir}ActiveCMs_NPIs.pt').rename('nRs', 'nDs', None), 
                  'ActiveCMs_wearing': t.load(f'{data_dir}ActiveCMs_wearing.pt').rename('nRs', 'nDs'), 
                  'ActiveCMs_mobility': t.load(f'{data_dir}ActiveCMs_mobility.pt').rename('nRs', 'nDs')}
    all_covariates = {'ActiveCMs_NPIs': t.load(f'{data_dir}ActiveCMs_NPIs_all.pt').rename('nRs', 'nDs', None).to(device), 
                      'ActiveCMs_wearing': t.load(f'{data_dir}ActiveCMs_wearing_all.pt').rename('nRs', 'nDs').to(device),
                      'ActiveCMs_mobility': t.load(f'{data_dir}ActiveCMs_mobility_all.pt').rename('nRs', 'nDs').to(device),}
    
    if not fake_data:
        data = {'obs': t.load(f'{data_dir}obs.pt').rename('nRs', 'nDs' ),}
        all_data = {'obs': t.load(f'{data_dir}obs_all.pt').rename('nRs', 'nDs' ).to(device),}

    else:
        P = get_P(all_platesizes, all_covariates)
        sample = P.sample()
        all_data = {'obs': sample.pop('obs').align_to('nRs', 'nDs' )}

        data = {'obs': all_data['obs'][:,:int(nDs*0.8)]}

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
    
    Expected_Log_Rs = lambda RegionR, CM_alpha, ActiveCMs_NPIs, Wearing_alpha, ActiveCMs_wearing, Mobility_alpha, ActiveCMs_mobility, prev: RegionR + \
                        CM_alpha@ActiveCMs_NPIs + Wearing_alpha*ActiveCMs_wearing + Mobility_alpha*ActiveCMs_mobility + prev

    P = Plate(
        #Effect of NPI
        CM_alpha = Normal(0, cm_prior_scale, sample_shape=[nCMs-2]),
        #Effect of mask wearing
        Wearing_alpha = Normal(wearing_mean, wearing_sigma),
        #Effect of mobility restrictions
        Mobility_alpha = Normal(mobility_mean, mobility_sigma),
        #R for each region
        RegionR = Normal(R_prior_mean_mean, R_prior_mean_scale + R_noise_scale),

        InitialSize_log_mean = Normal(math.log(1000), 0.5),
        log_infected_noise_mean = Normal(math.log(0.01), 0.25),
        nRs = Plate(
            #Initial number of infected in each region
            InitialSize_log = Normal(lambda InitialSize_log_mean: InitialSize_log_mean, 0.5),
            log_infected_noise = Normal(lambda log_infected_noise_mean: log_infected_noise_mean, 0.25),
            psi = Normal(math.log(1000), 1),
            nDs = Plate(
                log_infected = Timeseries('InitialSize_log', Normal(Expected_Log_Rs, lambda log_infected_noise: log_infected_noise.exp())),
                obs = NegativeBinomial(total_count=lambda psi: t.exp(psi), probs=lambda log_infected, psi: 1/((t.exp(psi)/ t.exp(log_infected)) + 1 + 1e-7) ),
            ),
        ),  
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    return P

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = get_P(platesizes, covariates)

    if Q_param_type == "opt":

        Q = Plate(
            npis = Group(
                CM_alpha = Normal(OptParam(t.ones((nCMs-2,))), OptParam(t.ones((nCMs-2,)), transformation=t.exp)),
                Wearing_alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                Mobility_alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                RegionR = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                InitialSize_log_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                log_infected_noise_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            ),
            nRs = Plate(
                    a = Group(
                        InitialSize_log = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                        log_infected_noise = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                        psi = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                    ),
                nDs = Plate(
                    log_infected = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                    obs = Data()
                ),
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates)

    else:
        assert Q_param_type == "qem"

        Q = Plate(
            npis = Group(
                CM_alpha = Normal(QEMParam(t.zeros((nCMs-2,))), QEMParam(t.ones((nCMs-2,)))),
                Wearing_alpha = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                Mobility_alpha = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                RegionR = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                InitialSize_log_mean = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                log_infected_noise_mean = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
            ),
            nRs = Plate(
                a = Group(
                        InitialSize_log = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                        log_infected_noise = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                        psi = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
                ),
                nDs = Plate(
                    log_infected = Normal(QEMParam(t.zeros(())), QEMParam(t.ones(()))),
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
    K = 3
    NUM_RUNS = 1
    NUM_ITERS = 100
    vi_lr = 0.1
    rws_lr = 0.03
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
        prob, all_data, all_covariates, all_platesizes = _load_and_generate_problem(device, 'opt')

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

            print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()

        print()
        print(f"RWS")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = _load_and_generate_problem(device, 'opt')

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

            print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()
            
        print()
        print(f"QEM")
        t.manual_seed(num_run)
        
        prob, all_data, all_covariates, all_platesizes = _load_and_generate_problem(device, 'qem')
        
        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)
            

        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_nograd()
            elbos['qem'][num_run, i] = elbo.detach()
            
            sample.update_qem_params(qem_lr)



            print(f"Iter {i}. Elbo: {elbo:.3f}")

