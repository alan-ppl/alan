import types
from typing import Optional
import inspect

import torch

from .utils import *
from .TorchDimDist import TorchDimDist
from .SamplingType import SamplingType

def func_args(something):
    """
    Takes an argument to the dist (either string, lambda or literal like `0`)
    """
    if isinstance(something, str):
        args = (something,)
        func = lambda scope: scope[something]
    elif isinstance(something, types.FunctionType):
        #The arguments to the function `something`.
        args = function_arguments(something)
        func = lambda scope: something(*[scope[arg] for arg in args])
    else:
        #something is e.g. 1 or 0.
        args = ()
        func = lambda scope: something
    return (args, func)


class Dist():
    """
    Abstract base class for distributions that are actually exposed to users.

    All the actual distributions (e.g. alan.Normal are subclasses of this distribution (the only difference 
    between e.g. alan.Normal and alan.Gamma is that the PyTorch distribution is stored on `self.dist`).

    These distributions are called by the user e.g. `alan.Normal(0, "a")` or `alan.Normal(0, lambda a: a.exp())`.

    There are three different types of argument we can give an Dist:
    A value (such as an integer/float).
    A string representing a name of a variable in-scope.
    A function representing a transformation of the scope.

    Critically, we extract the argument name from e.g. `lambda a: a.exp()` and use it to extract the right 
    variable from the scope.
    """
    def __init__(self, *args, sample_shape=t.Size([]), **kwargs):
        self.sample_shape = sample_shape

        #Converts args + kwargs to a unified dictionary mapping paramname2something,
        #following distributions initialization signature.
        paramname2something = inspect.signature(self.dist).bind(*args, **kwargs).arguments

        all_args = set()
        #A dict[str, function], where the functions map from a scope to a value.
        self.paramname2func = {}
        for paramname, something in paramname2something.items():
            args, func = func_args(something)
            self.paramname2func[paramname] = func
            all_args.update(args)

        self.all_args = list(all_args)

    def filter_scope(self, scope: dict[str, Tensor]):
        return {k: v for (k,v) in scope.items() if k in self.all_args}

    def tdd(self, scope: dict[str, Tensor]):
        paramname2val = {paramname: func(scope) for (paramname, func) in self.paramname2func.items()}
        return TorchDimDist(self.dist, **paramname2val)

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            inputs_params: dict,
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampling_type:SamplingType,
            reparam:bool):

        Kdim = groupvarname2Kdim[name]
        sample_dims = [Kdim, *active_platedims]

        filtered_scope = self.filter_scope(scope)
        resampled_scope = sampling_type.resample_scope(filtered_scope, active_platedims, Kdim)

        sample = self.tdd(resampled_scope).sample(reparam, sample_dims, self.sample_shape)

        return sample
    
    def sample_extended(
            self,
            sample:Tensor,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_original_platedims:list[Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            data:Optional[dict[str, Tensor]]):

        print(name)

        sample_dims = [Ndim, *active_extended_platedims]

        filtered_scope = self.filter_scope(scope)

        extended_sample = self.tdd(filtered_scope).sample(reparam, sample_dims, self.sample_shape)
        original_sample = sample if sample is not None else data[name]

        extended_sample = extended_sample.order(*active_extended_platedims)
        original_sample = original_sample.order(*active_original_platedims)

        original_idxs = [slice(0, dim.size) for dim in active_original_platedims]

        generic_setitem(extended_sample, original_idxs, original_sample)

        extended_sample = generic_getitem(extended_sample, active_extended_platedims)

        return extended_sample

    def log_prob(self, 
                 sample: Tensor, 
                 scope: dict[any, Tensor]):
        return self.tdd(scope).log_prob(sample)



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
    AD = type(name, (Dist,), {'dist': dist})
    globals()[name] = AD
    #setattr(alan, name, AD)

for dist in distributions:
    new_dist(dist, getattr(torch.distributions, dist))
