import torch as t
import numpy as np
from alan import Problem, Plate, BoundPlate, Normal, Timeseries, Data, QEMParam, OptParam
import matplotlib.pyplot as plt

def QQ(problem, num_samples, rvs, K, filename="QQ_plot.png"):
    # sample latents from the prior
    prior_latent_samples = problem.P.sample()
    prior_latent_sample_collection = {k: [v.rename(None)] for k,v in prior_latent_samples.items() if k in rvs}
    for _ in range(num_samples-1):
        prior_latent_sample_collection = {k: v + [problem.P.sample()[k].rename(None)] for k,v in prior_latent_sample_collection.items()}


    # sample latents from the posterior

    # this way doesn't seem to work...

    # posterior_latent_samples = problem.sample(K).importance_sample(1)
    # N_dim = posterior_latent_samples.Ndim
    # posterior_latent_sample_collection = {k: [posterior_latent_samples.samples_flatdict[k].order(N_dim).rename(None)] for k in prior_latent_sample_collection.keys()}
    # for _ in range(num_samples-1):
    #     posterior_latent_samples = problem.sample(K).importance_sample(1)
    #     N_dim = posterior_latent_samples.Ndim
    #     posterior_latent_sample_collection = {k: v + [posterior_latent_samples.samples_flatdict[k].order(N_dim).rename(None)] for k,v in posterior_latent_sample_collection.items()}

    posterior_latent_samples = problem.sample(K).importance_sample(num_samples)
    N_dim = posterior_latent_samples.Ndim
    posterior_latent_sample_collection = {k: [posterior_latent_samples.samples_flatdict[k].order(N_dim).rename(None)] for k in prior_latent_sample_collection.keys()}


    prior_latent_samples = {k: t.stack(v) for k,v in prior_latent_sample_collection.items()}
    posterior_latent_samples = {k: t.stack(v) for k,v in posterior_latent_sample_collection.items()}
                
    fig, ax = plt.subplots(1, len(rvs), figsize=(5*len(rvs), 5))
    if len(rvs) == 1:
        ax = [ax]
    for i, rv in enumerate(rvs):
        # sort the samples
        prior = prior_latent_samples[rv].sort()[0]
        posterior = posterior_latent_samples[rv].sort()[0]

        # plot the ordered samples
        ax[i].scatter(prior, posterior)
        ax[i].set_title(rv)
        ax[i].set_xlabel("Prior")
        ax[i].set_ylabel("Posterior")

        # also plot the line y=x
        lims = [
            np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
            np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
        ]
        ax[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.savefig(filename)


if __name__ == '__main__':
    K= 50

    P = Plate(
        mu = Normal(0., 1.), 
        p1 = Plate(
            z = Normal('mu', 1.),
            obs = Normal("z", 1.)
        )
    )
        
    Q = Plate(
        mu = Normal(QEMParam(0.), QEMParam(1.)),
        p1 = Plate(
            z = Normal('mu', 1.),
            obs = Data()
        )
    )


    platesizes = {'p1': 30}
    data = {'obs': t.randn((30,), names=('p1',))}

    P = BoundPlate(P, platesizes)
    Q = BoundPlate(Q, platesizes)

    prob = Problem(P, Q, data)

    QQ(prob, 1000, 'obs', ['mu'], K, "QQ_plot_pre_QEM.png")

    for _ in range(500):
        sample = prob.sample(K, True)
        sample.update_qem_params(0.3)
    
    
    QQ(prob, 1000, 'obs', ['mu'], K, "QQ_plot_post_QEM.png")

