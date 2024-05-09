import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import blackjax
import pickle
from collections import namedtuple

rng_key = jax.random.PRNGKey(0)

params = namedtuple("model_params", ["logvar", "obs_mean"])


N = 20
z_mean = 33
z_var = 0.5
obs_var = 10

loc, scale = 10, 20

with open('fake_data.pkl', 'rb') as f:
    platesizes, all_platesizes, data, all_data, _, _, _ = pickle.load(f)
observed = data['obs']
# observed = np.random.normal(loc, scale, size=(N,20))
# observed[:,1] += 10000
                            
def joint_logdensity(params):
    #prior
    # mean = stats.norm.logpdf(params.mean, z_mean, z_var)
    logvar = stats.norm.logpdf(params.logvar, 0., 1.)
    obs_mean = stats.norm.logpdf(params.obs_mean, 0., jnp.exp(params.logvar)).sum()
    # logobs_var = stats.norm.logpdf(params.logobs_var, params.mean, jnp.exp(params.logvar)).sum()
    obs = stats.norm.logpdf(observed, params.obs_mean, 1.).sum()
    
    return logvar + obs_mean +  obs
    

rng_key, init_key = jax.random.split(rng_key)


def init_param_fn(seed):
    """
    initialize a, b & thetas
    """
    key2, key3 = jax.random.split(seed, 2)
    return params(
        logvar=jax.random.normal(key2),
        obs_mean=jax.random.normal(key3, shape=(20,)),
    )


init_param = init_param_fn(init_key)
print(joint_logdensity(init_param))  # sanity check

warmup = blackjax.window_adaptation(blackjax.nuts, joint_logdensity)

# we use 4 chains for sampling
n_chains = 4
rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
init_keys = jax.random.split(init_key, n_chains)
init_params = jax.vmap(init_param_fn)(init_keys)

@jax.vmap
def call_warmup(seed, param):
    (initial_states, tuned_params), _ = warmup.run(seed, param, 1000)
    return initial_states, tuned_params

warmup_keys = jax.random.split(warmup_key, n_chains)
initial_states, tuned_params = jax.jit(call_warmup)(warmup_keys, init_params)

def inference_loop_multiple_chains(
    rng_key, initial_states, tuned_params, log_prob_fn, num_samples, num_chains
):
    kernel = blackjax.nuts.build_kernel()

    def step_fn(key, state, **params):
        return kernel(key, state, log_prob_fn, **params)

    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(step_fn)(keys, states, **tuned_params)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)

n_samples = 1000
rng_key, sample_key = jax.random.split(rng_key)
states, infos = inference_loop_multiple_chains(
    sample_key, initial_states, tuned_params, joint_logdensity, n_samples, n_chains
)

states = states.position._asdict()
#HMC means
HMC_means = {key: np.mean(states[key], axis=0).mean(0) for key in states}


#open means
with open('means.pkl', 'rb') as f:
    means = pickle.load(f)

#Differences
QEM_diffs = {key: [] for key in means}
for key in means:
    QEM_diffs[key] = ((np.expand_dims(HMC_means[key], 0) - means[key].reshape(250,-1)).mean(1)**2)
    
HMC_diffs = {key: [] for key in states}
for key in states:
    if key == 'logvar':
        HMC_diffs[key] = ((np.expand_dims(HMC_means[key], 0) - states[key].mean(1))**2)
    else:
        HMC_diffs[key] = ((np.expand_dims(HMC_means[key], 0) - states[key].mean(1)).mean(1)**2)




import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 2))

for key in QEM_diffs:
    ax[0].plot(QEM_diffs[key].tolist()[200:], label=f'{key}')
    ax[1].plot(HMC_diffs[key].tolist(), label=f'{key}')
    
ax[1].legend()
plt.show()

