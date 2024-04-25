import torch as t
import numpy as np
from alan import Problem, Plate, BoundPlate, Normal, Timeseries, Data, QEMParam, OptParam
import matplotlib.pyplot as plt

def QQ(problem, num_samples, data_name, rvs, K, filename="QQ_plot.png"):
    # sample latents from the prior
    prior_latent_samples = problem.P.sample()
    prior_latent_sample_collection = {k: [v.rename(None)] for k,v in prior_latent_samples.items() if k in rvs}
    for _ in range(num_samples-1):
        prior_latent_sample_collection = {k: v + [problem.P.sample()[k].rename(None)] for k,v in prior_latent_sample_collection.items()}


    # sample latents from the posterior

    temp_data = problem.P.sample().pop(data_name)[:,0]
    temp_problem = Problem(problem.P, problem.Q, {data_name: temp_data})
    posterior_latent_samples = temp_problem.sample(K).importance_sample(1)
    N_dim = posterior_latent_samples.Ndim
    posterior_latent_sample_collection = {k: [posterior_latent_samples.samples_flatdict[k].order(N_dim).rename(None)] for k in prior_latent_sample_collection.keys()}
    for _ in range(num_samples-1):
        temp_data = problem.P.sample().pop(data_name)[:,0]
        temp_problem = Problem(problem.P, problem.Q, {data_name: temp_data})
        posterior_latent_samples = problem.sample(K).importance_sample(1)
        N_dim = posterior_latent_samples.Ndim
        posterior_latent_sample_collection = {k: v + [posterior_latent_samples.samples_flatdict[k].order(N_dim).rename(None)] for k,v in posterior_latent_sample_collection.items()}

    prior_latent_samples = {k: t.stack(v) for k,v in prior_latent_sample_collection.items()}
    posterior_latent_samples = {k: t.stack(v) for k,v in posterior_latent_sample_collection.items()}
                
    fig, ax = plt.subplots(1, len(rvs), figsize=(5*len(rvs), 5))
    if len(rvs) == 1:
        ax = [ax]
    for i, rv in enumerate(rvs):
        # sort the samples
        prior = prior_latent_samples[rv].sort(0)[0]
        posterior = posterior_latent_samples[rv].sort(0)[0]

        # plot the ordered samples
        ax[i].scatter(prior, posterior)
        ax[i].set_title(f"QQ plot for {rv}")
        ax[i].set_xlabel("Latent drawn from p(z)")
        ax[i].set_ylabel("Latent drawn from p(z|x)")

        # also plot the line y=x
        lims = [
            np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
            np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
        ]
        ax[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.savefig(filename)


if __name__ == '__main__':
    K= 100

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


    platesizes = {'p1': 5}
    data = {'obs': t.randn((5,), names=('p1',))}

    P = BoundPlate(P, platesizes)

    # data = P.sample()['obs']
    Q = BoundPlate(Q, platesizes)

    prob = Problem(P, Q, data)

    QQ(prob, 500, 'obs', ['mu'], K, "QQ_plot_pre_QEM.png")

    for _ in range(100):
        sample = prob.sample(K, True)
        sample.update_qem_params(0.3)
    
    
    QQ(prob, 500, 'obs', ['mu'], K, "QQ_plot_post_QEM.png")

