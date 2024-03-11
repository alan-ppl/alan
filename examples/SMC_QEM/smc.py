import torch as t

def log_likelihood(ys, ts):
    return t.distributions.Normal(ts, 1).log_prob(ys)

def draw_sample_from_transition(ts, ts_log_var, K=1):
    return t.randn(K)*ts_log_var.exp() + ts*0.9

# get data
ts_init_true = t.tensor(1.5)
ts_log_var_true = t.tensor(0.9)

T = 5
ts = t.zeros(T)
ys = t.zeros(T)

ts[0] = draw_sample_from_transition(ts_init_true, ts_log_var_true)
ys[0] = ts[0] + t.randn(1)

for i in range(1, T):
    ts[i] = draw_sample_from_transition(ts[i-1], ts_log_var_true)
    ys[i] = ts[i] + t.randn(1)

def particle_filter(ts_init, ts_log_var, K=10, print_particles=False):
    # particle filter to infer the latent states and the model evidence

    particles = t.zeros(K, T)

    marginal_ll = t.zeros(())

    for i in range(T):
        prev = ts_init if i == 0 else particles[:, i-1]

        # sample K particles xs from the transition 
        xs = draw_sample_from_transition(ts_init, ts_log_var, K)
        particles[:, i] = xs

        # calculate the likelihood of the data given the particles
        weights = log_likelihood(ys[i], particles[:, i])

        marginal_ll += t.logsumexp(weights, 0) - t.tensor(K).log()

        # resample the particles
        resampled_indices = t.multinomial((weights - weights.amax()).exp(), K, replacement=True)
        particles = particles[resampled_indices, :]

        if print_particles:
            print(particles)

    # calculate the model evidence
    print(marginal_ll)


# prior
ts_init = t.tensor(1)
ts_log_var = t.tensor(1.3)

particle_filter(ts_init, ts_log_var, K=10)
particle_filter(ts_init, ts_log_var, K=100)

particle_filter(ts_init*10, ts_log_var*10, K=10)
particle_filter(ts_init*10, ts_log_var*10, K=100)

