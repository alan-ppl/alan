import torch as t
from alan import Normal, Bernoulli, HalfCauchy, Plate, BoundPlate, Group, Problem, Data, QEMParam, OptParam

def load_data_covariates(device, run=0, data_dir='data/'):
    # num_actors, num_actors_extended = 6, 7
    # num_blocks, num_blocks_extended = 4, 6
    num_actors, num_blocks = 7, 6
    num_repeats, num_repeats_extended = 10, 12

    platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks, 'plate_repeats': num_repeats}
    # all_platesizes = {'plate_actors': num_actors_extended, 'plate_blocks': num_blocks_extended}
    all_platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks, 'plate_repeats': num_repeats_extended}

    # platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks}
    # all_platesizes = {'plate_actors': num_actors, 'plate_blocks': num_blocks}

    data = {'obs':t.load(f'{data_dir}data_train.pt')}
    test_data = {'obs':t.load(f'{data_dir}data_test.pt')}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_actors','plate_blocks','plate_repeats')}
    
    data['obs'] = data['obs'].rename('plate_actors','plate_blocks','plate_repeats')


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

    # data['obs'] = data['obs'].to(device)
    # covariates['condition'] = covariates['condition'].to(device)
    # covariates['prosoc_left'] = covariates['prosoc_left'].to(device)

    # all_data['obs'] = all_data['obs'].to(device)
    # all_covariates['condition'] = all_covariates['condition'].to(device)
    # all_covariates['prosoc_left'] = all_covariates['prosoc_left'].to(device)

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def generate_problem(device, platesizes, data, covariates, Q_param_type):

    P = Plate(
        sigma_block = HalfCauchy(1.),
        sigma_actor = HalfCauchy(1.),

        beta_PC = Normal(0., 10.),
        beta_P = Normal(0., 10.),

        alpha = Normal(0., 10.),

        # plate_block = Plate(
        # ),

        # plate_actors = Plate(
        # ),

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

    if Q_param_type == "opt":
        Q = Plate(
            sigma_block = HalfCauchy(OptParam(1.)),
            sigma_actor = HalfCauchy(OptParam(1.)),

            # sigma_block = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),
            # sigma_actor = Normal(OptParam(0.), OptParam(0., transformation=t.exp)),

            beta_PC = Normal(OptParam(0.), OptParam(t.tensor(10.).log(), transformation=t.exp)),
            beta_P = Normal(OptParam(0.), OptParam(t.tensor(10.).log(), transformation=t.exp)),

            alpha = Normal(OptParam(0.), OptParam(t.tensor(10.).log(), transformation=t.exp)),

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
            sigma_block = HalfCauchy(OptParam(1.)),
            sigma_actor = HalfCauchy(OptParam(1.)),

            # sigma_block = Normal(QEMParam(0.), QEMParam(1.)),
            # sigma_actor = Normal(QEMParam(0.), QEMParam(1.)),

            beta_PC = Normal(QEMParam(0.), QEMParam(t.tensor(10.))),
            beta_P = Normal(QEMParam(0.), QEMParam(t.tensor(10.))),

            alpha = Normal(QEMParam(0.), QEMParam(t.tensor(10.))),

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

    K = 5

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

        # opt_P = t.optim.Adam(prob.Q.parameters(), lr=rws_lr, maximize=True)
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