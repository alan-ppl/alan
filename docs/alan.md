# Introduction to Alan

Alan is a probabilistic programming language that aims to provide a simple and intuitive way to perform fast Bayesian inference. In particular it is designed to showcase the "Massively Parallel" framework as outlined in [[Massively Parallel Reweighted Wake-Sleep]](https://arxiv.org/abs/2305.11022) and the source trick for computing posterior moments as outlined in [[Using autodiff to estimate posterior moments, marginals and samples]](https://arxiv.org/abs/2310.17374). This introduction assumes some prior familiarity with Bayesian inference and in particular variational Bayesian methods, for those who are not familiar with these [[Variational Inference: A Review for Statisticians]](https://arxiv.org/abs/1601.00670) gives a good overview.

## Specifying probabilistic models

Models are specified as a nested tree structure of `Plate`, `Group`, `Dist` and `Data` objects. The `Plate` object represents a plate, the `Group` object represents a group of variables that have their $K$ dimensions jointly sampled, `Dist` objects are probability distributions and the `Data` object represents a data tensor and is only used in approximate posteriors to ensure both $P$ and $Q$ contain the same variable names. The following example shows how to specify a simple generative model in Alan.


```python
P = Plate(
    ab = Group(
        a = Normal(0, 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("a", 1),
        p2 = Plate(
            e = Normal("d", 1.),
        ),
    ),
)
```

The above code defines a generative model with 5 variables, $a$, $b$, $c$, $d$ and $e$. The variables $a$ and $b$ are sampled together with the same $K$ dimension as they are grouped. The variable $c$ is sampled in a separate $K$ dimension to $a$ and $b$ and is dependent on $a$. The variable $e$ is dependent on $d$ which is in turn dependent on $a$. The corresponding approximate posterior is defined as follows:

```python
Q = Plate(
    ab = Group(
        a = Normal("a_mean", 1),
        b = Normal("a", 1),
    ),
    c = Normal(0, lambda a: a.exp()),
    p1 = Plate(
        d = Normal("d_mean", 1),
        p2 = Plate(
            e = Data()
        ),
    ),
)

Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})
```

We recommend structuring the approximate posterior in the same way as the generative model, however this is not required. The `BoundPlate` object is used to specify the parameters of the approximate posterior. The `Data` object is used to ensure that the approximate posterior contains the same variable names as the generative model.


We can then specify a problem by providing the generative model, approximate posterior, the plate sizes and the data.

```python
platesizes = {'p1': 3, 'p2': 4}
data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

prob = Problem(P, Q, platesizes, data)
```

## Sampling from probabilistic models

We can then sample from the generative model using the `sample` method. This method takes the number of samples to generate, the type of sampling to use and whether to use reparameterisation. The following example shows how to generate a sample from the generative model with $K=5$ using the independent sampling method.


```python
# Get some initial samples (with K dims)
K=5
sampling_type = IndependentSample
sample = prob.sample(K, reparam=True, sampling_type=sampling_type)
```


## Sampling Marginals, Conditionals, K's from the posterior and computing moments

We can also sample from the marginals ($P(k)$) and conditionals $P(K_c|K_a)$ of the generative model using the `sample.marginals()` and `sample.conditionals()` methods respectively. These return a dictionary of samples for each variable. 

```python
conditionals = sample.conditionals()
marginals = sample.marginals()
```

We can also sample from the posterior $P(K|data)$ using the `sample.sample_posterior(num_samples)` method. This returns a dictionary of sampled $K$ indices for each variables.

```python
# Obtain K indices from posterior
post_idxs = sample.sample_posterior(num_samples=10)
```

These sampled posterior indices can be used to compute moments of the posterior using the `sample.moments()` method. This returns a dictionary of moments for each variable.

```python
# Create posterior samples explicitly using sample and post_idxs
isample = IndexedSample(sample, post_idxs)

def mean(x):
    sample = x
    dim = x.dims[-1]

    w = 1/dim.size
    return (w * sample).sum(dim)

def second_moment(x):
    return mean(t.square(x))

def square(x):
    return x**2

def var(x):
    return mean(square(x)) - square(mean(x))

moments = sample.moments({'d': [mean, var], 'c': [second_moment]}, post_idxs, isample)
```

As you see we must pass as arguments a dictionary of functions to compute the moments for each variable, the sampled posterior indices and the `IndexedSample` object. 

