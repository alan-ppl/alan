import torch as t
from alan import Normal, Binomial, Plate, BoundPlate, Group, Problem, Data, QEMParam

def load_data_covariates(device, run=0, data_dir='data/'):
    M, J, I = 3, 3, 30
    platesizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}
    all_platesizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':2*I}

    data = {'obs':t.load(f'{data_dir}delay_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    test_data = {'obs':t.load(f'{data_dir}delay_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    covariates = {'run_type': t.load(f'{data_dir}run_type_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load(f'{data_dir}bus_company_name_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.load(f'{data_dir}run_type_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load(f'{data_dir}bus_company_name_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat((covariates['run_type'],test_covariates['run_type']),2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],2)}
    

    # data['obs'] = data['obs'].to(device)
    # covariates['run_type'] = covariates['run_type'].to(device)
    # covariates['bus_company_name'] = covariates['bus_company_name'].to(device)

    # all_data['obs'] = all_data['obs'].to(device)
    # all_covariates['run_type'] = all_covariates['run_type'].to(device)
    # all_covariates['bus_company_name'] = all_covariates['bus_company_name'].to(device)

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def generate_problem(device, platesizes, data, covariates, Q_param_type):
    M, J, I = 3, 3, 30

    bus_company_name_dim = covariates['bus_company_name'].shape[-1]
    run_type_dim = covariates['run_type'].shape[-1]

    P = Plate(
        log_sigma_phi_psi = Normal(0, 1),

        psi = Normal(t.zeros((run_type_dim,)), t.ones((run_type_dim,))),
        phi = Normal(t.zeros((bus_company_name_dim,)), t.ones((bus_company_name_dim,))),

        sigma_beta = Normal(0, 1),
        mu_beta = Normal(0, 1),

        plate_Year = Plate(
            beta = Normal('mu_beta', lambda sigma_beta: sigma_beta.exp()),

            sigma_alpha = Normal(0, 1),

            plate_Borough = Plate(
                alpha = Normal('beta', lambda sigma_alpha: sigma_alpha.exp()),
        
                plate_ID = Plate(
                    obs = Binomial(total_count=131, logits = lambda alpha, phi, psi, run_type, bus_company_name: alpha + phi @ bus_company_name + psi @ run_type),
                )
            )
        )

    )

    P = BoundPlate(P, platesizes, inputs = covariates)

    if Q_param_type == "opt":

        Q = Plate(
            log_sigma_phi_psi = Normal("log_sigma_phi_psi_loc", "log_sigma_phi_psi_scale"),

            psi = Normal("psi_loc", "psi_scale"),
            phi = Normal("phi_loc", "phi_scale"),

            sigma_beta = Normal("sigma_beta_loc", "sigma_beta_scale"),
            mu_beta = Normal("mu_beta_loc", "mu_beta_scale"),

            plate_Year = Plate(
                beta = Normal("beta_loc", "beta_scale"),

                sigma_alpha = Normal("sigma_alpha_loc", "sigma_alpha_scale"),

                plate_Borough = Plate(
                    alpha = Normal("alpha_loc", "alpha_scale"),

                    plate_ID = Plate(
                        obs = Data()
                    )
                )
            )
        )

        Q = BoundPlate(Q, platesizes, inputs = covariates,
                        extra_opt_params = {"log_sigma_phi_psi_loc":   t.zeros(()),
                                            "log_sigma_phi_psi_scale": t.ones(()),

                                            "psi_loc":   t.zeros((run_type_dim,)),
                                            "psi_scale": t.ones((run_type_dim,)),
                                            "phi_loc":   t.zeros((bus_company_name_dim,)),
                                            "phi_scale": t.ones((bus_company_name_dim,)),
                                            
                                            "sigma_beta_loc":   t.zeros(()), 
                                            "sigma_beta_scale": t.ones(()),
                                            "mu_beta_loc":      t.zeros(()), 
                                            "mu_beta_scale":    t.ones(()),

                                            "beta_loc":         t.zeros((M,), names=('plate_Year',)),
                                            "beta_scale":       t.ones((M,),  names=('plate_Year',)),

                                            "sigma_alpha_loc":   t.zeros((M,), names=('plate_Year',)),
                                            "sigma_alpha_scale": t.ones((M,), names=('plate_Year',)),
                                            
                                            "alpha_loc":         t.zeros((M,J,), names=('plate_Year','plate_Borough')),
                                            "alpha_scale":       t.ones((M,J,),  names=('plate_Year','plate_Borough'))})

    else:
        assert Q_param_type == "qem"

        Q = Plate(
            log_sigma_phi_psi = Normal(QEMParam(0.), QEMParam(1.)),

            psi = Normal(t.zeros((run_type_dim,)), t.ones((run_type_dim,))),
            phi = Normal(t.zeros((bus_company_name_dim,)), t.ones((bus_company_name_dim,))),

            sigma_beta = Normal(QEMParam(0.), QEMParam(1.)),
            mu_beta = Normal(QEMParam(0.), QEMParam(1.)),

            plate_Year = Plate(
                beta = Normal(QEMParam(0.), QEMParam(1.)),

                sigma_alpha = Normal(QEMParam(0.), QEMParam(1.)),

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
        opt = torchopt.Adam(prob.Q.parameters(), lr=0.005)

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
        plt.savefig('plots/quick_elbos.png')

        if DO_PREDLL:
            plt.figure()
            plt.plot(t.arange(NUM_ITERS), lls['vi'].mean(0), label='VI')
            plt.plot(t.arange(NUM_ITERS), lls['rws'].mean(0), label='RWS')
            plt.plot(t.arange(NUM_ITERS), lls['qem'].mean(0), label='QEM')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('PredLL')
            plt.savefig('plots/quick_predlls.png')