import torch as t
from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample, Data

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

platesizes = {'plate_1': M, 'plate_2': N}
all_platesizes = {'plate_1': M, 'plate_2': 2*N}

covariates = {'x':t.load(f'data/weights_{N}_{M}.pt')}
test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}.pt')}
all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

P = BoundPlate(P, inputs = covariates)

Q = BoundPlate(Q, inputs = covariates,
                  params = {"mu_z_loc":   t.zeros((d_z,)), 
                            "mu_z_scale": t.ones((d_z,)),
                            "psi_z_loc":   t.zeros((d_z,)), 
                            "psi_z_scale": t.ones((d_z,)),
                            "z_loc":   t.zeros((M, d_z), names=('plate_1', None)),
                            "z_scale": t.ones((M, d_z), names=('plate_1', None))})

data = {'obs':t.load(f'data/data_y_{N}_{M}.pt')}
test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}.pt')}
all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
data['obs'] = data['obs'].rename('plate_1','plate_2')
all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')

prob = Problem(P, Q, platesizes, data)

sampling_type = IndependentSample

K = 3
opt = t.optim.Adam(prob.Q.parameters(), lr=0.01)
for i in range(100):
    opt.zero_grad()

    sample = prob.sample(K, True, sampling_type)
    elbo = sample.elbo()

    (-elbo).backward()
    opt.step()

    ll = sample.predictive_ll(all_platesizes=all_platesizes, reparam=True, all_data=all_data, all_inputs=all_covariates)
    print(f"Iter {i}. Elbo: {elbo:.3f}, PredLL: {ll['obs']:.3f}")
