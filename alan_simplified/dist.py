def valid_input(x, active_platedims):
    """
    It only makes sense to use some inputs (specifically, we can't use inputs which have plates
    which aren't currently active).
    """
    return all((dim in active_platedims) for dim in generic_dims(x))

def filtered_inputs(inputs, active_platedims):
    return {name: inp for name, inp in inputs.items() if valid_input(inp, active_platedims)}


class AlanDist():
    """
    Distribution class that is actually exposed to users.

    Takes only kwargs.

    kwargs can be two things: a string, or a lambda.

    Called as Distribution(Normal, loc="a", scale=lambda tr: tr["b"].exp(), shape=t.Size([]))
    """
    def __init__(self, *, sample_shape=t.Size([]), **kwargs):
        self.sample_shape = shape

        #Sugar, converting Distribution(Normal, loc="a", scale="b")
        #to loc=lambda tr: tr["a"] , scale=lambda tr["b"]
        for name, arg in kwargs.items():
            if isinstance(arg, str):
                kwargs[arg] = lambda tr: tr[arg]

        #A dict of lambdas that take tr as input.
        self.kwargs_func = kwargs

    def torchdimdist(self, parent_trace, current_trace, inputs, active_platedims):
        inputs = filtered_inputs(inputs, active_platedims)

        #Construct unified trace.
        tr = {**parent_trace, **current_trace, **inputs}
        #Check that there are no overlapping keys in any of the dicts
        assert len(tr) == len(parent_trace) + len(current_trace) + len(inputs)

        kwargs = {name: func(tr) for (name, func) in self.kwargs_func}

        return TorchDimDist(self.dist, **kwargs)

    def sample(self, name, inputs, active_platedims: List[str], all_platedims: Dict[str, Dim], K=None, reparam=True):

        tdd = self.torchdimdist(parent_trace, current_trace, inputs, active_platedims)

        return tdd.sample(reparam, sample_dims, sample_shape)

    def log_prob(self, 



distributions = [
"Bernoulli",
"Beta",
"Binomial",
"Categorical",
"Cauchy",
"Chi2",
"ContinuousBernoulli",
"Dirichlet",
"Exponential",
"FisherSnedecor",
"Gamma",
"Geometric",
"Gumbel",
"HalfCauchy",
"HalfNormal",
"Kumaraswamy",
"LKJCholesky",
"Laplace",
"LogNormal",
"LowRankMultivariateNormal",
"Multinomial",
"MultivariateNormal",
"NegativeBinomial",
"Normal",
"Pareto",
"Poisson",
"RelaxedBernoulli",
"RelaxedOneHotCategorical",
"StudentT",
"Uniform",
"VonMises",
"Weibull",
"Wishart",
]

def new_dist(name, dist):
    """
    This is the function called by external code to add a new distribution to Alan.
    Arguments:
        name: string, will become the class name for the distribution.
        dist: Distribution class mirroring standard PyTorch distribution API.
    """
    AD = type(name, (AlanDist,), {'dist': dist})
    globals()[name] = AD
    setattr(alan, name, AD)

for dist in distributions:
    new_dist(dist, getattr(torch.distributions, dist))
