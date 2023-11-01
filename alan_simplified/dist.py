import types
import inspect

import torch

from .utils import *
from .TorchDimDist import TorchDimDist

def in_plate(x, active_platedims: list[Dim], all_platedims: dict[str, Dim]):
    """
    It only makes sense to use some inputs (specifically, we can't use inputs which have plates
    which aren't currently active).
    """
    non_active_platedims = set(all_platedims.values()).difference(active_platedims)
    return all((dim not in non_active_platedims) for dim in generic_dims(x))

def function_arguments(f):
    """
    Extracts the arguments of f as a list of strings.

    Does lots of error checking to ensure the function signature is very simple
    (e.g. no *args, no **kwargs, no default args, no kw-only args, no type annotations)
    """
    argspec = inspect.getfullargspec(f)

    #no *args
    if argspec.varargs is not None:
        raise Exception("In Alan, functions may not have *args")
    #no **kwargs
    if argspec.varkw is not None:
        raise Exception("In Alan, functions may not have **kwargs")
    #no defaults (positional or keyword only)
    if (argspec.defaults is not None) or (argspec.kwonlydefaults is not None):
        raise Exception("In Alan, functions may not have defaults")
    #no keyword only arguments
    if argspec.kwonlyargs:
        #kwonlyargs is a list, and lists evaluate to False if empty
        raise Exception("In Alan, functions may not have keyword only arguments")
    #no type annotations
    if argspec.annotations:
        #Annotations is a dict, and dicts evaluate to False if empty
        raise Exception("In Alan, functions may not have type annotations")

    return argspec.args

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


class AlanDist():
    """
    Abstract base class for distributions that are actually exposed to users.

    All the actual distributions (e.g. Alan.Normal are subclasses of this distribution (the only difference 
    between e.g. alan.Normal and alan.Gamma is that the PyTorch distribution is stored on `self.dist`).

    These distributions are called by the user e.g. `alan.Normal(0, "a")` or `alan.Normal(0, lambda a: a.exp())`.

    There are three different types of argument we can give an AlanDist:
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

    def filter_scope(self, scope, active_platedims, all_platedims):
        """
        Filters down to only the variables in scope that are actually used.

        Two criteria:
        1 Filter out the variables that aren't actually used in this distribution.  
          That's so that when we do e.g. permutations, we're only permuting the variables we need to.
        2 Filter out "input" variables from lower-level plates.
        """
        result = {}
        for varname in self.all_args:
            if varname not in scope:
                raise Exception(f"Can't find {varname} in scope")
            tensor = scope[varname]
            if not in_plate(tensor, active_platedims, all_platedims):
                raise Exception(f"{varname} is at a lower plate-level: it has dimensions {tensor.dims}, while at the moment, we're in the plate with dims {active_platedims}")
            result[varname] = tensor
        return result

    def sample(self, 
               scope: dict[str, Tensor], 
               active_platedims: list[str], 
               all_platedims: dict[str, Dim], 
               sampling_type,
               Kdim=None, 
               reparam=True):

        scope = self.filter_scope(scope, active_platedims, all_platedims)

        paramname2val = {paramname: func(scope) for (paramname, func) in self.paramname2func.items()}
        tdd = TorchDimDist(self.dist, **paramname2val)

        sample_dims = [Kdim, *active_platedims]
        return tdd.sample(reparam, sample_dims, self.sample_shape)

    def log_prob(self, 
                 sample: Tensor, 
                 scope: dict[any, Tensor], 
                 active_platedims: list[str], 
                 all_platedims: dict[str, Dim], 
                 sampling_type,
                 groupvarname2Kdim:dict[str, Dim]):
        """
        Not enough information here to apply sampling_type summing over dimensions!!!
        """

        scope = self.filter_scope(scope, active_platedims, all_platedims)

        paramname2val = {paramname: func(scope) for (paramname, func) in self.paramname2func.items()}
        tdd = TorchDimDist(self.dist, **paramname2val)
        return tdd.log_prob(sample)



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
    #setattr(alan, name, AD)

for dist in distributions:
    new_dist(dist, getattr(torch.distributions, dist))
