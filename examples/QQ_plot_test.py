import torch as t
import numpy as np
from alan import Problem, Plate, BoundPlate, Normal, Timeseries, Data, QEMParam, OptParam, mean, mean2
import matplotlib.pyplot as plt

def QQ(problem, num_samples, data_name, rvs, K, filename="QQ_plot.png"):
    # sample latents from the prior
    prior_latent_samples = problem.P.sample()
    prior_latent_sample_collection = {k: [v.rename(None)] for k,v in prior_latent_samples.items() if k in rvs}

    # sample latents from the posterior
    temp_problem = Problem(problem.P, problem.Q, {data_name: prior_latent_samples[data_name][:,0]})
    posterior_latent_samples = temp_problem.sample(K).importance_sample(1)
    N_dim = posterior_latent_samples.Ndim
    posterior_latent_sample_collection = {k: [posterior_latent_samples.samples_flatdict[k].order(N_dim).rename(None)] for k in prior_latent_sample_collection.keys()}
    
    # repeat the process for the remaining n-1 samples
    for _ in range(num_samples-1):
        prior_latent_samples = problem.P.sample()
        prior_latent_sample_collection = {k: v + [prior_latent_samples[k].rename(None)] for k,v in prior_latent_sample_collection.items()}

        temp_problem = Problem(problem.P, problem.Q, {data_name: prior_latent_samples[data_name][:,0]})
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
    K= 300

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
    # data = {'obs': t.randn((5,), names=('p1',))}

    P = BoundPlate(P, platesizes)

    # data = P.sample()['obs']
    Q_bound = BoundPlate(Q, platesizes)

    # QQ(prob, 500, 'obs', ['mu'], K, "QQ_plot_pre_QEM.png")

    # for _ in range(100):
    #     sample = prob.sample(K, True)
    #     sample.update_qem_params(0.3)
    
    
    # QQ(prob, 500, 'obs', ['mu'], K, "QQ_plot_post_QEM.png")
    
    num_samples = 1000
    prior_latent_samples = P.sample(num_samples)
    
    data_samples = prior_latent_samples['obs'].rename(None).rename('p1', None)
    print(data_samples.var())
    
    print('mu')
    print(prior_latent_samples['mu'].mean())
    print(prior_latent_samples['mu'].var())
    
    print('z')
    print(prior_latent_samples['z'].mean(1))
    print(prior_latent_samples['z'].var(1))
    
    mu_means = []
    mu_mean2s = []
    mu_vars = []
    
    z_means = []
    z_mean2s = []
    z_vars = []
    
    post_mu = []
    post_z = []
    for i in range(num_samples):
        Q = Plate(
        mu = Normal(QEMParam(0.), QEMParam(1.)),
        p1 = Plate(
            z = Normal('mu', 1.),
            obs = Data()
        )
        )
        Q_bound = BoundPlate(Q, platesizes)
        prob = Problem(P, Q_bound, {'obs': data_samples[:,i]})
        # for _ in range(50):
        #     sample = prob.sample(K, reparam=False)
        #     sample.update_qem_params(0.3)
        
        posterior_samples = prob.sample(K)
        posterior_latent_samples = posterior_samples.importance_sample(1)
        N_dim = posterior_latent_samples.Ndim
        dims = posterior_latent_samples.samples_flatdict['z'].dims
        post_mu.append(posterior_latent_samples.samples_flatdict['mu'].order(N_dim).rename(None))
        post_z.append(posterior_latent_samples.samples_flatdict['z'].order(dims[::-1]).rename(None))
        posterior_means = posterior_samples.moments([('mu', mean), ('mu', mean2), ('z', mean), ('z', mean2)])
        
        mu_means.append(posterior_means[0])
        mu_vars.append(posterior_means[1] - posterior_means[0]**2)
        
        z_means.append(posterior_means[2].rename(None))
        z_vars.append(posterior_means[3].rename(None) - posterior_means[2].rename(None)**2)
        
        mu_mean2s.append(posterior_means[1])
        z_mean2s.append(posterior_means[3].rename(None))
        
        
    
    print('mu')
    print(t.stack(mu_means).mean())
    print(t.stack(mu_means).var() + t.stack(mu_vars).mean())
    print(t.stack(mu_mean2s).mean())
    
    print('z')
    print(t.stack(z_means).mean(0))
    print(t.stack(z_means).var(0) + t.stack(z_vars).mean(0))
    print(t.stack(z_mean2s).mean(0))
    
    # print(t.stack(mu_mean2s) - t.stack(mu_means)**2)
    print(t.stack(mu_means))
    print(t.stack(mu_means).var())
    # print(t.stack(z_mean2s))
    
    
    

    #plot means against each other with diagonal line
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(prior_latent_samples['mu'].rename(None).sort(0)[0], t.stack(post_mu).sort(0)[0])
    # also plot the line y=x
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlabel("Latent drawn from p(z)")
    ax.set_ylabel("Latent drawn from p(z|x)")
    plt.savefig('QQ_test_mu.png')
    
        #plot means against each other with diagonal line
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        ax[i].scatter(prior_latent_samples['z'][i,:].rename(None).sort(0)[0], t.stack(post_z)[:,i].sort(0)[0])
        # also plot the line y=x
        lims = [
            np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
            np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
        ]
        ax[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax[i].set_xlabel("Latent drawn from p(z)")
        
    ax[0].set_ylabel("Latent drawn from p(z|x)")
    plt.savefig('QQ_test_z.png')