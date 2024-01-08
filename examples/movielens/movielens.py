import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, Data, QEMParam

def load_data_covariates(device, run=0, data_dir='data/'):
    M, N = 300, 5

    platesizes = {'plate_1': M, 'plate_2': N}
    all_platesizes = {'plate_1': M, 'plate_2': 2*N}

    data = {'obs':t.load(f'{data_dir}data_y_{N}_{M}.pt')}
    test_data = {'obs':t.load(f'{data_dir}test_data_y_{N}_{M}.pt')}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')
    all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')

    covariates = {'x':t.load(f'{data_dir}weights_{N}_{M}.pt')}
    test_covariates = {'x':t.load(f'{data_dir}test_weights_{N}_{M}.pt')}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

    # data['obs'] = data['obs'].to(device)
    # covariates['x'] = covariates['x'].to(device)

    # all_data['obs'] = all_data['obs'].to(device)
    # all_covariates['x'] = all_covariates['x'].to(device)

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    d_z = 18
    M, N = 300, 5

    P = Plate(
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

    if Q_param_type == "opt":
        Q = Plate(
            mu_z = Normal("mu_z_loc", "mu_z_scale"),
            psi_z = Normal("psi_z_loc", "psi_z_scale"),

            plate_1 = Plate(
                z = Normal("z_loc", "z_scale"),

                plate_2 = Plate(
                    obs = Data()
                )
            ),
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates,
                        extra_opt_params = {"mu_z_loc":   t.zeros((d_z,)), 
                                            "mu_z_scale": t.ones((d_z,)),
                                            "psi_z_loc":   t.zeros((d_z,)), 
                                            "psi_z_scale": t.ones((d_z,)),
                                            "z_loc":   t.zeros((M, d_z), names=('plate_1', None)),
                                            "z_scale": t.ones((M, d_z), names=('plate_1', None))})

    else:
        assert Q_param_type == 'qem'

        Q = Plate(
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

def load_and_generate_problem(device, Q_param_type, run=0, data_dir='data/'):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run, data_dir)
    problem = generate_problem(device, platesizes, data, covariates, Q_param_type)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    import torchopt
    DO_PLOT   = True
    DO_PREDLL = False
    NUM_ITERS = 250
    NUM_RUNS  = 3

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

        K = 3
        # opt = t.optim.Adam(prob.Q.parameters(), lr=0.01)
        opt = torchopt.Adam(prob.Q.parameters(), lr=0.01)
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

        K = 3
        # opt_P = t.optim.Adam(prob.Q.parameters(), lr=0.01)
        opt = torchopt.Adam(prob.Q.parameters(), lr=0.01, maximize=True)

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

        K = 3
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

            sample.update_qem_params(0.1)

    if DO_PLOT:
        for key in elbos.keys():
            elbos[key] = elbos[key].cpu()
            lls[key] = lls[key].cpu()

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t.arange(NUM_ITERS), elbos['vi'].mean(0), label='VI')
        plt.plot(t.arange(NUM_ITERS), elbos['rws'].mean(0), label='RWS')
        plt.plot(t.arange(NUM_ITERS), elbos['qem'].mean(0), label='QEM')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title('Movielens')
        plt.savefig('plots/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label='VI')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label='RWS')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label='QEM')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.title('Movielens')
            plt.savefig('plots/quick_predlls.png')