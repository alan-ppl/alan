import types
from typing import Optional
import inspect
import warnings

import torch

from .utils import *
from .TorchDimDist import TorchDimDist
from .Sampler import Sampler
from .Stores import BufferStore
from .Param import QEMParam, OptParam, Param
from .Data import Data

def datagroup(group):
    assert isinstance(group, dict)
    hasdata = any(isinstance(v, Data) for v in group.values())
    more_than_one = 2 <= len(group)
    assert not ((more_than_one) and hasdata)
    return hasdata


def sample_gdt(
        prog:dict,
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        K_dim: Dim,
        groupvarname2Kdim,
        sampler:Sampler,
        reparam:bool,
        ):

    assert not datagroup(prog)



    #All arguments on prog
    set_all_arg_list = set([arg for dist in prog.values() for arg in dist.all_args])
    #Remove references to self (either for previous things in the Group, or timeseries refs to self).
    all_args = set_all_arg_list.difference([*prog.keys(), 'prev']) #remove dependencies on other variables in the group.

    #Used mainly for non-timeseries
    sample_dims = [K_dim, *active_platedims] #Don't sample T_dim.

    #Used mainly for timeseries
    if 1<=len(active_platedims):
        other_platedims, T_dim = (active_platedims[:-1], active_platedims[-1])

    result = {}       #This is the returned samples.

    for k in all_args:
        if k not in scope:
            raise Exception(f"{k} is not in scope") #!!!

    #Filter scope, to include only variables that are actually used.
    #Just for efficiency.
    scope = {k: v for (k,v) in scope.items() if k in all_args}

    #Resample K-dimensions; returns scope variables with just K_dim.
    #also does this with the initial state.
    scope = sampler.resample_scope(scope, active_platedims, K_dim)

    #Permutation to resample across timesteps
    timeseries_perm = sampler.perm(dims=set(sample_dims), Kdim=K_dim)

    for name, dist in prog.items():
        sample = dist.sample(scope, reparam, active_platedims, K_dim, timeseries_perm)

        scope[name]  = sample
        result[name] = sample

    return result



def apply_func_val(func_val, scope):
    if isinstance(func_val, str):
        return scope[func_val]
    elif isinstance(func_val, types.FunctionType):
        args = function_arguments(func_val)
        return func_val(*[scope[arg] for arg in args])
    else:
        assert isinstance(func_val, (Number, Tensor))
        return func_val

class _Dist:
    def __init__(self, *args, sample_shape=t.Size([]), **kwargs):
        self.args = args
        self.sample_shape = sample_shape
        self.kwargs = kwargs

        if len(args) + len(kwargs) != self.nargs:
            raise Exception(f"Wrong number of arguments provided to {type(self)}")

    def finalize(self, varname):
        return Dist(
            varname=varname, 
            dist=self.dist, 
            args=self.args, 
            sample_shape=self.sample_shape, 
            kwargs=self.kwargs
        )

class Dist(torch.nn.Module):
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
    def __init__(self, varname, dist, args, sample_shape, kwargs):
        super().__init__()
        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.is_timeseries = False

        self.dist = dist

        self.register_buffer("_device_tensor", t.zeros(()))

        self.sample_shape = sample_shape

        #Dict mapping distargname (e.g. loc and scale in a Normal) to:
        #OptParam
        #QEMParam
        #str
        #Number
        #Named Tensor.
        #Lambda
        distargname2func_val_param = inspect.signature(self.dist).bind(*args, **kwargs).arguments

        #QEM Error checking:
        #  If any parameter is QEM, then all parameters must be QEM
        #  All QEM parameters must have the same ignore_platedims
        self.qem_dist = 0 < sum(isinstance(x, QEMParam) for x in distargname2func_val_param.values())
        if self.qem_dist:
            func_val_params = list(distargname2func_val_param.values())
            for func_val_param in func_val_params:
                if not isinstance(func_val_param, QEMParam):
                    raise Exception("If one parameter on a distribution is a QEMParam, then all parameters on that distribution should be QEM distributions")

            set_ignore_platenames0 = set(func_val_params[0].ignore_platenames)
            for func_val_param in func_val_params[1:]:
                if set_ignore_platenames0 != set(func_val_param.ignore_platenames):
                    raise Exception("If one parameter on a distribution is a QEMParam, then all parameters on that distribution should be QEM distributions")

        #Dict mapping distargname (e.g. loc and scale in a Normal) to:
        #str
        #Number
        #Named Tensor.
        #Lambda
        #Converted OptParam + QEMParam to strings + saved them to opt_params or qem_params.
        distargname2func_val = {}
        self.opt_qem_params = {}
        for distargname, func_val_param in distargname2func_val_param.items():
            if isinstance(func_val_param, Param):
                if varname is None:
                    raise Exception(f"You can't use QEMParam / OptParam in a timeseries at present")
                name = func_val_param.name if (func_val_param.name is not None) else f"{varname}_{distargname}"
                self.opt_qem_params[name] = (distargname, func_val_param)
                func_val_param = name
            distargname2func_val[distargname] = func_val_param

            


        all_args = set()

        self.str_args = {}
        self.func_args = {}
        tensor_args = {}
        self.val_args = {}

        for distargname, func_val in distargname2func_val.items():
            if isinstance(func_val, str):
                self.str_args[distargname] = func_val
                all_args.update((func_val,))
            elif isinstance(func_val, types.FunctionType):
                self.func_args[distargname] = func_val
                all_args.update(function_arguments(func_val))
            elif isinstance(func_val, torch.Tensor):
                tensor_args[distargname] = func_val
            else:
                print(distargname, func_val)
                assert isinstance(func_val, Number)
                self.val_args[distargname] = func_val

        self.tensor_args = BufferStore(tensor_args)
        self.all_args = list(all_args)

    @property
    def device(self):
        return self._device_tensor.device

    def filter_scope(self, scope: dict[str, Tensor]):
        return {k: v for (k,v) in scope.items() if k in self.all_args}

    def paramname2val(self, scope: dict[str, Tensor]):
        result = {}

        for paramname, val in self.val_args.items():
            result[paramname] = self.move_number_device(paramname, val)
        for paramname, tensor in self.tensor_args.to_dict().items():
            assert tensor.device == self.device
            result[paramname] = tensor
        for paramname, arg in self.str_args.items():
            result[paramname] = scope[arg]
        for paramname, func in self.func_args.items():
            val = func(*[scope[arg] for arg in function_arguments(func)])
            if not isinstance(val, Tensor):
                raise Exception("Lambda on a distribution returned a non-Tensor")
            if val.device != self.device:
                raise Exception(f"Lambda on a distribution returned a tensor on the wrong device.  Expected a tenson on {self.device}, whereas we got a tensor on {val.device}")
            result[paramname] = val

        return result

    def tdd(self, scope: dict[str, Tensor]):
        return TorchDimDist(self.dist, **self.paramname2val(scope))
    
    def sample_extended(
            self,
            sample:Tensor,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:dict):

        filtered_scope = self.filter_scope(scope)

        sample_dims = [*active_extended_platedims, Ndim]
                    
        original_sample = sample if sample is not None else original_data[name]

        tdd = self.tdd(filtered_scope)
        extended_sample = tdd.sample(reparam, sample_dims, self.sample_shape)

        # Need to ensure that we work with lists of platedims in corresponding orders for original and extended samples.
        original_dims, extended_dims = corresponding_plates(original_platedims, extended_platedims, original_sample, extended_sample)

        original_sample = generic_order(original_sample, original_dims)
        extended_sample = generic_order(extended_sample, extended_dims)

        # Insert the original sample into the extended sample
        original_idxs = [slice(0, dim.size) for dim in original_dims]
        generic_setitem(extended_sample, original_idxs, original_sample)

        # Put extended_dims back on extended_sample
        extended_sample = generic_getitem(extended_sample, extended_dims)

        return extended_sample

    def predictive_ll(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            original_data:dict[str, Tensor],
            extended_data:dict[str, Tensor]):
        
        original_ll, extended_ll = {}, {}

        if name in extended_data.keys():
            extended_ll[name], _ = self.log_prob(extended_data[name], scope, None, None)

            original_dims, extended_dims = corresponding_plates(original_platedims, extended_platedims, original_data[name], extended_data[name]) 

            # Take the logprob of the original data from the extended logprob tensor
            original_idxs = [slice(0, dim.size) for dim in original_dims]
            original_ll[name] = generic_getitem(generic_order(extended_ll[name], extended_dims), original_idxs)
            original_ll[name] = generic_getitem(original_ll[name], original_dims)

        return original_ll, extended_ll


    def log_prob(self, sample: Tensor, scope: dict, T_dim, K_dim):
        #T_dim, Kinit_dim, K_dim are deliberately ignored; they're only there so that 
        #interface matches Timeseries.

        assert self.device == sample.device
        return self.tdd(scope).log_prob(sample), None

    def sample(self, scope, reparam, active_platedims, K_dim, timeseries_perm=None):
        return self.tdd(scope).sample(
            reparam=reparam, 
            sample_dims=[*active_platedims, K_dim],
            sample_shape=self.sample_shape,
        )

    def move_number_device(self, param_name, param):
        assert isinstance(param, Number)
        #If its not a tensor, check dist is expecting a float param
        float_param = not self.dist.arg_constraints[param_name].is_discrete

        if float_param:
            param=float(param)
        return t.tensor(param, device=self.device)




distributions_nargs = [
("Bernoulli", 1),
("Beta", 2),
("Binomial", 2),
("Categorical", 1),
("Cauchy", 2),
("Chi2", 1),
("ContinuousBernoulli", 1),
("Dirichlet", 1),
("Exponential", 1),
("FisherSnedecor", 2),
("Gamma", 2),
("Geometric", 1),
("Gumbel", 2),
("HalfCauchy", 1),
("HalfNormal", 1),
("Kumaraswamy", 2),
("LKJCholesky", 2),
("Laplace", 2),
("LogNormal", 2),
("LowRankMultivariateNormal", 3),
("Multinomial", 2),
("MultivariateNormal", 2),
("NegativeBinomial", 2),
("Normal", 2),
("OneHotCategorical", 1),
("Pareto", 2),
("Poisson", 1),
("RelaxedBernoulli", 2),
("LogitRelaxedBernoulli", 2),
("RelaxedOneHotCategorical", 2),
("StudentT", 3),
("Uniform", 2),
("VonMises", 2),
("Weibull", 2),
("Wishart", 2),
]

def new_dist(name, dist, nargs):
    """
    This is the function called by external code to add a new distribution to Alan.
    Arguments:
        name: string, will become the class name for the distribution.
        dist: Distribution class mirroring standard PyTorch distribution API.
    """
    AD = type(name, (_Dist,), {'dist': dist, 'nargs': nargs})
    globals()[name] = AD
    #setattr(alan, name, AD)

for dist, nargs in distributions_nargs:
    if hasattr(torch.distributions, dist):
        new_dist(dist, getattr(torch.distributions, dist), nargs)
