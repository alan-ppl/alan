import torch as t
from alan import Normal, Binomial, Bernoulli, ContinuousBernoulli, Uniform, Beta, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam

def load_data_covariates(device, run=0, data_dir='data/', fake_data=False):
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

        # breakpoint()

    else:
        print("Sampling fake data")
        P = get_P(all_platesizes, all_covariates)
        all_data = {'obs': P.sample()['obs'].align_to('plate_Years', 'plate_Birds', 'plate_Ids', 'plate_Replicate')}

        data = {'obs': all_data['obs'][:,:,:I,:]}

        # breakpoint()

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
                bird_year_mean = Normal('bird_mean', 1.), # how common is this bird this year?

                plate_Ids = Plate(
                    
                    # z = Binomial(total_count=10, logits=lambda weather, bird_year_mean, beta: bird_year_mean*weather*beta), # how many of this bird were actually present?
                    z = Bernoulli(logits=lambda weather, bird_year_mean, beta: bird_year_mean*weather*beta), # was this bird actually present?

                    plate_Replicate = Plate(
                        # obs = Binomial(total_count=10, logits=lambda alpha, quality, z: alpha * quality * z) # how many of this bird did we actually see?
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
            bird_mean_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
            bird_mean_log_var = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),

            alpha_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
            alpha_log_var = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),

            beta_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),
            beta_log_var = Normal(OptParam(0.), OptParam(0., transformation=t.exp),),

            plate_Birds = Plate(
                bird_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how common is this bird?

                alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how easy is this bird to see?

                beta = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how much does weather affect this bird?

                plate_Years = Plate(
                    bird_year_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp),), # how common is this bird this year?

                    plate_Ids = Plate(
                        
                        # z = Binomial(total_count=10, logits=lambda weather, bird_year_mean, beta: bird_year_mean*weather*beta), # how many of this bird were actually present?
                        z = Bernoulli(logits=lambda weather, bird_year_mean, beta: bird_year_mean*weather*beta), # was this bird actually present?

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
            bird_mean_mean = Normal(OptParam(0.), OptParam(1.),),
            bird_mean_log_var = Normal(OptParam(0.), OptParam(1.),),

            alpha_mean = Normal(OptParam(0.), OptParam(1.),),
            alpha_log_var = Normal(OptParam(0.), OptParam(1.),),

            beta_mean = Normal(OptParam(0.), OptParam(1.),),
            beta_log_var = Normal(OptParam(0.), OptParam(1.),),

            plate_Birds = Plate(
                bird_mean = Normal(OptParam(0.), OptParam(1.),), # how common is this bird?

                alpha = Normal(OptParam(0.), OptParam(1.),), # how easy is this bird to see?

                beta = Normal(OptParam(0.), OptParam(1.),), # how much does weather affect this bird?

                plate_Years = Plate(
                    bird_year_mean = Normal(OptParam(0.), OptParam(1.),), # how common is this bird this year?

                    plate_Ids = Plate(
                        
                        # z = Binomial(total_count=10, logits=lambda weather, bird_year_mean, beta: bird_year_mean*weather*beta), # how many of this bird were actually present?
                        z = Bernoulli(logits=lambda weather, bird_year_mean, beta: bird_year_mean*weather*beta), # was this bird actually present?

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
    # fake_data = True
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir, fake_data)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)

    # if splitting on Replicates not on Ids:
    # all_covariates = covariates
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    DO_PLOT   = True
    DO_PREDLL = True
    NUM_ITERS = 5
    NUM_RUNS  = 1

    K = 3

    vi_lr = 0.1
    rws_lr = 0.1
    qem_lr = 0.1

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # device='cpu'

    FAKE_DATA = False

    if FAKE_DATA:
        def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
            return _load_and_generate_problem(device, Q_param_type, run, data_dir, fake_data=True)
        
    else:
        def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
            return _load_and_generate_problem(device, Q_param_type, run, data_dir, fake_data=False)

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
        # print(f"VI")
        # t.manual_seed(num_run)
        # prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        # for key in all_data.keys():
        #     all_data[key] = all_data[key].to(device)
        # for key in all_covariates.keys():
        #     all_covariates[key] = all_covariates[key].to(device)

        # opt = t.optim.Adam(prob.Q.parameters(), lr=vi_lr)
        # opt = torchopt.Adam(prob.Q.parameters(), lr=vi_lr)
        # for i in range(NUM_ITERS):
        #     opt.zero_grad()

        #     sample = prob.sample(K, True)
        #     elbo = sample.elbo_vi()
        #     elbos['vi'][num_run, i] = elbo.detach()

        #     if DO_PREDLL:
        #         importance_sample = sample.importance_sample(N=10)
        #         extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
        #         ll = extended_importance_sample.predictive_ll(all_data)
        #         lls['vi'][num_run, i] = ll['obs']
        #         print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
        #     else:
        #         print(f"Iter {i}. Elbo: {elbo:.3f}")

        #     (-elbo).backward()
        #     opt.step()

        print()
        print(f"RWS")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        opt = t.optim.Adam(prob.Q.parameters(), lr=rws_lr, maximize=True)
        # opt = torchopt.Adam(prob.Q.parameters(), lr=rws_lr, maximize=True)

        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, False)
            elbo = sample.elbo_rws()
            elbos['rws'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=3)
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
            sample = prob.sample(K, False)
            elbo = sample.elbo_nograd()
            elbos['qem'][num_run, i] = elbo

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=3)
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
        # plt.plot(t.arange(NUM_ITERS), elbos['vi'].mean(0), label=f'VI lr={vi_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['rws'].mean(0), label=f'RWS lr={rws_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['qem'].mean(0), label=f'QEM lr={qem_lr}')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title(f'Occupancy (K={K}, extending on Ids, {"FAKE" if FAKE_DATA else "REAL"} data)')
        plt.savefig('plots/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            # plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'Occupancy (K={K}, extending on Ids, {"FAKE" if FAKE_DATA else "REAL"} data)')
            plt.savefig('plots/quick_predlls.png')