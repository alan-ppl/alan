import torch as t
from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample

d_z = 18
M, N = 300, 5

P = Plate(
    mu_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),
    psi_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),

    # mu_z = Normal(t.zeros(()), t.ones(()), sample_shape=(d_z,)),
    # psi_z = Normal(t.zeros(()), t.ones(()), sample_shape=(d_z,)),

    plate_1 = Plate(
        z = Normal("mu_z", lambda psi_z: psi_z.exp()),

        plate_2 = Plate(
            obs = Bernoulli(logits = lambda z, x: z @ x),
        )
    ),
)

Q = Plate(
    # mu_z = Normal("mu_z_loc", "mu_z_scale", sample_shape=(d_z,)),
    # psi_z = Normal("psi_z_loc", "psi_z_scale", sample_shape=(d_z,)),

    # plate_1 = Plate(
    #     # z = Normal("mu_z", "psi_z"),
    #     z = Normal("z_loc", "z_scale", sample_shape=(d_z,)),

    #     plate_2 = Plate(
    #     )
    # ),

    mu_z = Normal("mu_z_loc", "mu_z_scale"),
    psi_z = Normal("psi_z_loc", "psi_z_scale"),

    plate_1 = Plate(
        z = Normal("z_loc", "z_scale"),

        plate_2 = Plate(
        )
    ),
)

platesizes = {'plate_1': M, 'plate_2': N}
all_platesizes = {'plate_1': 2*M, 'plate_2': 2*N}

covariates = {'x':t.load(f'data/weights_{N}_{M}.pt')}
test_covariates = {'x':t.load(f'data/test_weights_{N}_{M}.pt')}
all_covariates = {'x': t.cat([covariates['x'],test_covariates['x']],-2).rename('plate_1','plate_2',...)}
covariates['x'] = covariates['x'].rename('plate_1','plate_2',...)
all_covariates['x'] = all_covariates['x'].rename('plate_1','plate_2',...)

Q = BoundPlate(Q, inputs = covariates,
                  params = {"mu_z_loc":   t.zeros((d_z,)), 
                            "mu_z_scale": t.ones((d_z,)),
                            "psi_z_loc":   t.zeros((d_z,)), 
                            "psi_z_scale": t.ones((d_z,)),
                            "z_loc":   t.zeros((M, d_z), names=('plate_1', None)),
                            "z_scale": t.ones((M, d_z), names=('plate_1', None))})
                #   params = {"mu_z_loc":   t.zeros(()), 
                #             "mu_z_scale": t.ones(()),
                #             "psi_z_loc":   t.zeros(()), 
                #             "psi_z_scale": t.ones(()),
                #             "z_loc":   t.zeros((M,), names=('plate_1',)),
                #             "z_scale": t.ones((M,), names=('plate_1',))})

data = {'obs':t.load(f'data/data_y_{N}_{M}.pt')}
test_data = {'obs':t.load(f'data/test_data_y_{N}_{M}.pt')}
all_data = {'obs': t.cat([data['obs'],test_data['obs']], -1).rename('plate_1','plate_2')}
data['obs'] = data['obs'].rename('plate_1','plate_2')
all_data['obs'] = all_data['obs'].rename('plate_1','plate_2')

prob = Problem(P, Q, platesizes, data)

sampling_type = IndependentSample
sample = prob.sample(3, True, sampling_type)
print(sample.elbo())


K_samples = sample.sample_posterior(num_samples=10)

print(K_samples)