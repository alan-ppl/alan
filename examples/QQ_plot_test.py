import torch as t
import numpy as np
from alan import Problem, Plate, BoundPlate, Normal, Timeseries, Data, QEMParam, OptParam, mean, mean2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    K= 3z00

    prior_mean = 20
    prior_scale = 6
    prior_var = prior_scale**2
    prior_prec = 1/prior_var

    z_scale = 30
    obs_scale = 4

    like_var = z_scale**2 + obs_scale**2
    like_prec = 1/like_var

    N = 10
    data = 1.5+t.randn(N)
    post_prec = prior_prec + data.shape[0]*like_prec
    post_mean = (prior_prec*prior_mean + like_prec*data.sum()) / post_prec

    P = Plate(
        mu = Normal(prior_mean, prior_scale), 
        p1 = Plate(
            z = Normal(lambda mu: mu, z_scale),
            obs = Normal("z", obs_scale)
        )
    )
        
    Q = Plate(
        mu = Normal(QEMParam(0.), QEMParam(4.)),
        p1 = Plate(
            z = Normal(0., 3.5),
            obs = Data()
        )
    )

    known_moments = {
        ('mu', mean): post_mean,
        ('mu', mean2): post_mean**2 + 1/post_prec,
    }

    platesizes = {'p1': N}
    # data = {'obs': t.randn((5,), names=('p1',))}

    P = BoundPlate(P, platesizes)

    # data = P.sample()['obs']
    Q_bound = BoundPlate(Q, platesizes)
    
    num_samples = 500
    prior_latent_samples = P.sample(num_samples)
    
    data_samples = prior_latent_samples['obs'].rename(None).rename('p1', None)
    
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
            mu = Normal(QEMParam(0.), QEMParam(0.)),
            p1 = Plate(
                z = Normal(QEMParam(0.), QEMParam(1.)),
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
    print('mean')
    print(t.stack(mu_means).mean())
    print('var')
    print(t.stack(mu_means).var() + t.stack(mu_vars).mean())
    print('vars')
    


    
    print('z')
    print('mean')
    print(t.stack(z_means).mean(0))
    print('var')
    print(t.stack(z_means).var(0) + t.stack(z_vars).mean(0))


    
    # # print(t.stack(mu_mean2s) - t.stack(mu_means)**2)
    # print(t.stack(mu_means))
    # print(t.stack(mu_means).var())
    # # print(t.stack(z_mean2s))

    

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