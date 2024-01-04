import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, Data

def load_data_covariates(device, run=0):
    M, N = 300, 5

    platesizes = {'plate_1': M, 'plate_2': N}
    all_platesizes = {'plate_1': M, 'plate_2': 2*N}

    data = {'obs':t.load(f'data/data_y_{N}_{M}.pt')}
    test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}.pt')}
    all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
    data['obs'] = data['obs'].rename('plate_1','plate_2')
    all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')

    covariates = {'x':t.load(f'data/weights_{N}_{M}.pt')}
    test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}.pt')}
    all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
    covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
    all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

    data['obs'] = data['obs'].to(device)
    covariates['x'] = covariates['x'].to(device)

    all_data['obs'] = all_data['obs'].to(device)
    all_covariates['x'] = all_covariates['x'].to(device)

    return platesizes, all_platesizes, data, all_data, covariates, all_covariates

def generate_problem(device, platesizes, data, covariates):
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

    P = BoundPlate(P, platesizes, inputs = covariates)

    Q = BoundPlate(Q, platesizes, inputs = covariates,
                      extra_opt_params = {"mu_z_loc":   t.zeros((d_z,)), 
                                          "mu_z_scale": t.ones((d_z,)),
                                          "psi_z_loc":   t.zeros((d_z,)), 
                                          "psi_z_scale": t.ones((d_z,)),
                                          "z_loc":   t.zeros((M, d_z), names=('plate_1', None)),
                                          "z_scale": t.ones((M, d_z), names=('plate_1', None))})

    prob = Problem(P, Q, data)
    prob.to(device)

    return prob

def load_and_generate_problem(device, run=0):
    platesizes, all_platesizes, data, all_data, covariates, all_covariates = load_data_covariates(device, run)
    problem = generate_problem(device, platesizes, data, covariates)
    
    return problem, all_data, all_covariates, all_platesizes

if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    prob, all_data, all_covariates, all_platesizes = load_and_generate_problem(device)

    K = 3
    opt = t.optim.Adam(prob.Q.parameters(), lr=0.01)
    for i in range(10):
        opt.zero_grad()

        sample = prob.sample(K, True)
        elbo = sample.elbo_vi()

        importance_sample = sample.importance_sample(N=10)
        extended_importance_sample = importance_sample.extend(all_platesizes, extended_inputs=all_covariates)
        ll = extended_importance_sample.predictive_ll(all_data)
        print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")

        (-elbo).backward()
        opt.step()
