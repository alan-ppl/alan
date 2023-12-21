import torch as t
from alan_simplified import Normal, Binomial, Plate, BoundPlate, Group, Problem, IndependentSample, Data


def generate_model(device, run=0):
    M = 3
    J = 3
    I = 30

    platesizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':I}
    all_platesizes = {'plate_Year': M, 'plate_Borough':J, 'plate_ID':2*I}

    
    data = {'obs':t.load(f'data/delay_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    test_data = {'obs':t.load(f'data/delay_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...)}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']],-1)}

    covariates = {'run_type': t.load(f'data/run_type_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load(f'data/bus_company_name_train_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    test_covariates = {'run_type': t.load(f'data/run_type_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float(),
        'bus_company_name': t.load(f'data/bus_company_name_test_{run}.pt').rename('plate_Year', 'plate_Borough', 'plate_ID',...).float()}
    all_covariates = {'run_type': t.cat((covariates['run_type'],test_covariates['run_type']),2),
        'bus_company_name': t.cat([covariates['bus_company_name'],test_covariates['bus_company_name']],2)}

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

    P = BoundPlate(P, inputs = covariates)

    Q = BoundPlate(Q, inputs = covariates,
                      params = {"log_sigma_phi_psi_loc":   t.zeros(()),
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


    return Problem(P, Q, platesizes, data), all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    prob, all_data, all_covariates, all_platesizes = generate_model('cpu')

    sampling_type = IndependentSample
    K = 3
    opt = t.optim.Adam(prob.Q.parameters(), lr=0.01)
    for i in range(100):
        opt.zero_grad()

        sample = prob.sample(K, True, sampling_type)
        elbo = sample.elbo()

        importance_sample = sample.importance_sample(num_samples=10)
        extended_importance_sample = importance_sample.extend(all_platesizes, False, extended_inputs=all_covariates)
        ll = extended_importance_sample.predictive_ll(all_data)
        print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")

        (-elbo).backward()
        opt.step()