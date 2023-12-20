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
            reparam:bool,
            device:torch.device,
            ):

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
            original_data:Optional[dict[str, Tensor]],
            extended_data:Optional[dict[str, Tensor]]):

        filtered_scope = self.filter_scope(scope)

        # dicts to store logprobs of original and extended data (if needed)
        original_ll = {}
        extended_ll = {}

        if sample is not None or extended_data is None: 
            # "sample is not None" means this dist is not for observed data, so we need to sample it
            # "extended_data is None" means even if this dist *is* observed, we should sample it, not return a logprob

            sample_dims = [*active_extended_platedims, Ndim]
                        
            original_sample = sample if sample is not None else original_data[name]
            extended_sample = self.tdd(filtered_scope).sample(reparam, sample_dims, self.sample_shape)

            # Need to ensure that we work with lists of platedims in corresponding orders for original and extended samples.
            original_dims, extended_dims = corresponding_plates(original_platedims, extended_platedims, original_sample, extended_sample)

            original_sample = generic_order(original_sample, original_dims)
            extended_sample = generic_order(extended_sample, extended_dims)

            # Insert the original sample into the extended sample
            original_idxs = [slice(0, dim.size) for dim in original_dims]
            generic_setitem(extended_sample, original_idxs, original_sample)

            # Put extended_dims back on extended_sample
            extended_sample = generic_getitem(extended_sample, extended_dims)

            # Put Ndim back at end of dims. (not sure if this is necessary)
            # if Ndim in set(extended_sample.dims):
            #     extended_sample = extended_sample.order(Ndim)[Ndim]

            return extended_sample, original_ll, extended_ll

        else:
            # Put extended_platedims onto extended_data (which is a dict of tensors without Dims)
            # Note that original_data already has the correct Dims due to it being passed through Problem()
            extended_data_with_dims = extended_data[name].rename(None)[active_extended_platedims]

            extended_ll[name] = self.tdd(filtered_scope).log_prob(extended_data_with_dims)

            original_dims, extended_dims = corresponding_plates(original_platedims, extended_platedims, original_data[name], extended_data_with_dims) 

            # Take the logprob of the original data from the extended logprob tensor
            original_idxs = [slice(0, dim.size) for dim in original_dims]
            original_ll[name] = generic_getitem(generic_order(extended_ll[name], extended_dims), original_idxs)
            original_ll[name] = generic_getitem(original_ll[name], original_dims)

            return extended_data[name], original_ll, extended_ll


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
