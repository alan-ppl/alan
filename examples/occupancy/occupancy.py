import torch as t
from alan import Normal, Bernoulli, ContinuousBernoulli, Uniform, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam

def load_data_covariates(device, run=0, data_dir='data/'):
    M, J, I, Returns = 6, 12, 200, 5
    I_extended = 300

    platesizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I, 'plate_Replicate': Returns}
    all_platesizes = {'plate_Years': M, 'plate_Birds':J, 'plate_Ids':I_extended, 'plate_Replicate': Returns}

    data = {'obs':t.load(f'{data_dir}birds_train_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}
    test_data = {'obs':t.load(f'{data_dir}birds_test_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids','plate_Replicate')}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-2)}

    covariates = {'weather': t.load(f'{data_dir}weather_train_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load(f'{data_dir}quality_train_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    test_covariates = {'weather': t.load(f'{data_dir}weather_test_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float(),
        'quality': t.load(f'{data_dir}quality_test_{run}.pt').rename('plate_Years', 'plate_Birds', 'plate_Ids').float()}
    all_covariates = {'weather': t.cat([covariates['weather'],test_covariates['weather']],-1),
        'quality': t.cat([covariates['quality'],test_covariates['quality']],-1)}
    
    data['obs'] = data['obs'].float()
    all_data['obs'] = all_data['obs'].float()

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = Plate(
        plate_Years = Plate(
            year_mean = Normal(0., 1.),

            plate_Birds = Plate(
                bird_mean = Normal('year_mean', 1.),

                plate_Ids = Plate(
                    beta = Normal('bird_mean', 1.),
                    
                    # u = Uniform(0., 1.),
                    z = ContinuousBernoulli(logits=lambda weather, bird_mean: bird_mean*weather),

                    alpha = Normal('bird_mean', 1.),

                    plate_Replicate = Plate(
                        obs = Bernoulli(logits=lambda alpha, quality, z: alpha * quality * z)
                        # obs = Bernoulli(logits=lambda alpha, quality, bird_mean, weather, u: alpha * quality * (bird_mean*weather + t.log(u/1-u)))
                    )
                ),
            )
        )
    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    if Q_param_type == "opt":
        Q = Plate(
            plate_Years = Plate(
                year_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                plate_Birds = Plate(
                    bird_mean = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

                    plate_Ids = Plate(
                        beta = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
                        
                        # z = ContinuousBernoulli(logits=lambda weather, bird_mean: bird_mean*weather),
                        z = ContinuousBernoulli(OptParam(t.tensor(0.5).log(), transformation=t.exp)),

                        alpha = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

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
            plate_Years = Plate(
                year_mean = Normal(QEMParam(0.), QEMParam(1.)),

                plate_Birds = Plate(
                    bird_mean = Normal(QEMParam(0.), QEMParam(1.)),

                    plate_Ids = Plate(
                        beta = Normal(QEMParam(0.), QEMParam(1.)),
                        
                        # z = ContinuousBernoulli(logits=lambda weather, bird_mean: bird_mean*weather),
                        z = ContinuousBernoulli(QEMParam(0.5)),

                        alpha = Normal(QEMParam(0.), QEMParam(1.)),

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

def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    import torchopt
    DO_PLOT   = True
    DO_PREDLL = True
    NUM_ITERS = 100
    NUM_RUNS  = 1

    K = 3

    vi_lr = 0.1
    rws_lr = 0.1
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
        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        opt = t.optim.Adam(prob.Q.parameters(), lr=vi_lr)
        # opt = torchopt.Adam(prob.Q.parameters(), lr=vi_lr)
        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_vi()
            elbos['vi'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
                extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
                ll = extended_importance_sample.predictive_ll(all_data)
                lls['vi'][num_run, i] = ll['obs']
                print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
            else:
                print(f"Iter {i}. Elbo: {elbo:.3f}")

            (-elbo).backward()
            opt.step()

        print()
        print(f"RWS")
        t.manual_seed(num_run)

        prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device, 'opt')

        for key in all_data.keys():
            all_data[key] = all_data[key].to(device)
        for key in all_covariates.keys():
            all_covariates[key] = all_covariates[key].to(device)

        # opt = t.optim.Adam(prob.Q.parameters(), lr=rws_lr, maximize=True)
        opt = torchopt.Adam(prob.Q.parameters(), lr=rws_lr, maximize=True)

        for i in range(NUM_ITERS):
            opt.zero_grad()

            sample = prob.sample(K, True)
            elbo = sample.elbo_rws()
            elbos['rws'][num_run, i] = elbo.detach()

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
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
            sample = prob.sample(K, True)
            elbo = sample.elbo_nograd()
            elbos['qem'][num_run, i] = elbo

            if DO_PREDLL:
                importance_sample = sample.importance_sample(N=10)
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
        plt.plot(t.arange(NUM_ITERS), elbos['vi'].mean(0), label=f'VI lr={vi_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['rws'].mean(0), label=f'RWS lr={rws_lr}')
        plt.plot(t.arange(NUM_ITERS), elbos['qem'].mean(0), label=f'QEM lr={qem_lr}')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title(f'Movielens (K={K})')
        plt.savefig('plots/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label=f'VI lr={vi_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label=f'RWS lr={rws_lr}')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label=f'QEM lr={qem_lr}')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title(f'Movielens (K={K})')
            plt.savefig('plots/quick_predlls.png')