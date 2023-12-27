import random 
import string

from typing import Optional, Union, List

from torch.autograd import grad
from functorch.dim import Dim

from .Sampler import Sampler
from .Split import Split
from .Plate import Plate, tensordict2tree, flatten_tree, empty_tree
from .utils import *
from .logpq import logPQ_plate
from .sample_logpq import logPQ_sample
from .BoundPlate import BoundPlate
from .Marginals import Marginals
from .ImportanceSample import ImportanceSample
from .Split import Split, no_checkpoint, checkpoint
from .moments import RawMoment, torchdim_moments_mixin, named_moments_mixin


class Sample():
    def __init__(
            self,
            problem,
            sample: dict,
            groupvarname2Kdim: dict[str, Dim],
            sampler: Sampler,
            reparam: bool,
        ):
        self.problem = problem
        self.sample = sample
        self.groupvarname2Kdim = groupvarname2Kdim
        self.sampler = sampler
        self.reparam = reparam

    @property
    def device(self):
        return self.problem.device

    @property 
    def P(self):
        return self.problem.P

    @property 
    def Q(self):
        return self.problem.Q

    @property 
    def all_platedims(self):
        return self.problem.all_platedims

    def _elbo(self, extra_log_factors, computation_strategy):
        if extra_log_factors is None:
            extra_log_factors = {}

        #Sum out any non-torchdim dimensions.
        extra_log_factors = {k: sum_non_dim(v) for (k, v) in extra_log_factors.items()}

        for v in extra_log_factors.values():
            #I have very little idea why this line is necessary ... but without it, 
            #we get `RuntimeError: Attempt to retrieve a tensor saved by 
            #autograd multiple times without checkpoint recomputation being 
            #triggered in between, this is not currently supported. Please 
            #open an issue with details on your use case so that we can 
            #prioritize adding this.` if we compute a latent variable for a 
            #moment we have split.
            #
            #my guess is it ensures the variable is part of the compute graph, which
            #gets forgotten otherwise
            v.clone()

        #extra_log_factors comes in as a flat dict; convert to tree.
        extra_log_factors = tensordict2tree(self.P.plate, extra_log_factors)

        lp = logPQ_plate(
            name=None,
            P=self.P.plate, 
            Q=self.Q.plate, 
            sample=self.sample,
            inputs_params=self.problem.inputs_params(),
            data=self.problem.data,
            extra_log_factors=extra_log_factors,
            scope={}, 
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=self.groupvarname2Kdim,
            sampler=self.sampler,
            computation_strategy=computation_strategy)

        return lp

    def elbo_vi(self, computation_strategy=checkpoint):
        if not self.reparam==True:
            raise Exception("To compute the ELBO with the right gradients for VI you must construct a reparameterised sample using `problem.sample(K, reparam=True)`")
        return self._elbo(extra_log_factors=None, computation_strategy=computation_strategy)

    def elbo_rws(self, computation_strategy=checkpoint):
        if not self.reparam==False:
            raise Exception("To compute the ELBO with the right gradients for RWS you must construct a non-reparameterised sample using `problem.sample(K, reparam=False)`")
        return self._elbo(extra_log_factors=None, computation_strategy=computation_strategy)

    def elbo_nograd(self, computation_strategy=checkpoint):
        if not self.reparam==False:
            raise Exception("elbo_nograd has no gradients, so you should construct a non-reparameterised sample using `problem.sample(K, reparam=False)`")
        with t.no_grad():
            result = self._elbo(extra_log_factors=None, computation_strategy=computation_strategy)
        return result
    
    def _importance_sample_idxs(self, N:int, computation_strategy):
        """
        User-facing method that returns reweighted samples.
        """

        #extra_log_factors doesn't make sense for posterior sampling, but is required for
        #one of the internal methods.
        extra_log_factors = empty_tree(self.P.plate)
        assert isinstance(extra_log_factors, dict)

        N_dim = Dim('N', N)
        
        with t.no_grad():
            indices = logPQ_sample(
                name=None,
                P=self.P.plate, 
                Q=self.Q.plate, 
                sample=self.sample,
                inputs_params=self.problem.inputs_params(),
                data=self.problem.data,
                extra_log_factors=extra_log_factors,
                scope={}, 
                active_platedims=[],
                all_platedims=self.all_platedims,
                groupvarname2Kdim=self.groupvarname2Kdim,
                sampler=self.sampler,
                computation_strategy=computation_strategy,
                indices={},
                num_samples=N,
                N_dim=N_dim,
            )

        Kdim2groupvarname = {v: k for (k, v) in self.groupvarname2Kdim.items()}
        assert len(Kdim2groupvarname) == len(self.groupvarname2Kdim)

        indices = {Kdim2groupvarname[k]: v for (k, v) in indices.items()}
        return indices, N_dim

    def importance_sample(self, N:int, computation_strategy=checkpoint):
        """
        User-facing method that returns reweighted samples.
        """
        indices, N_dim = self._importance_sample_idxs(N=N, computation_strategy=computation_strategy)

        samples = index_into_sample(self.sample, indices, self.groupvarname2Kdim, self.P.varname2groupvarname())

        return ImportanceSample(self.problem, samples, N_dim)

    def _marginal_idxs(self, joints, computation_strategy):
        """
        Internal method that returns a flat dict mapping frozenset describing the K-dimensions in the marginal to a Tensor.
        """

        for joint in joints:
            if not isinstance(joint, tuple):
                raise Exception("Arguments to marginals must be a tuple of groupvarnames, representing joint marginal to evaluate")

            if len(joint) < 2:
                raise Exception("Arguments to marginals must be a tuple of groupvarnames of length 2 or above (as we're doing all the univariate marginals anyway")

            for groupvarname in joint:
                if not groupvarname in self.groupvarname2Kdim:
                    raise Exception("Arguments provided to marginals must be groupvarnames, not varnames.  Specifically, if there's a variable in a Group, you should provide the name of the Group, not the name of the variable")

        univariates = tuple(frozenset([varname]) for varname in self.groupvarname2Kdim.keys())
        joints = tuple(frozenset(joint) for joint in joints)

        joints = univariates + joints

        #List of named Js to go into torch.autograd.grad
        J_tensor_list = []
        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
        J_torchdim_dict = {}
        #dimension names
        dimss = []

        groupvarname2active_platedimnames = self.problem.P.groupvarname2active_platedimnames()

        for groupvarnames_frozenset in joints:
            #Convert frozenset groupvarnames to tuple.
            groupvarnames = tuple(groupvarnames_frozenset)

            #Check that all variables are part of the same plate.
            active_platedimnames = groupvarname2active_platedimnames[groupvarnames[0]]
            set_active_platedimnames = set(active_platedimnames)
            for groupvarname in groupvarnames[:1]:
                if set_active_platedimnames != set(groupvarname2active_platedimnames[groupvarname]):
                    raise Exception("Trying to compute marginal for variables at different plates; not sure this makes sense")

            active_platedims = [self.all_platedims[dimname] for dimname in active_platedimnames]
            
            Kdims = [self.groupvarname2Kdim[groupvarname] for groupvarname in groupvarnames]

            dims = [*Kdims, *active_platedims]
            dimss.append(dims)
            shape = [dim.size for dim in dims]

            J_tensor = t.zeros(shape, device=self.device, requires_grad=True)
            J_tensor_list.append(J_tensor)
            J_torchdim = generic_getitem(J_tensor, dims)
            
            J_torchdim_dict[groupvarnames_frozenset] = J_torchdim

        #Compute loss
        L = self._elbo(extra_log_factors=J_torchdim_dict, computation_strategy=computation_strategy)
        #marginals as a list
        marginals_list = grad(L, J_tensor_list)

        result = {}
        for gvn, dims, marginals in zip(joints, dimss, marginals_list):
            result[gvn] = marginals[dims]

        return result

    def marginals(self, *joints, computation_strategy=checkpoint):
        """
        User-facing method that returns a marginals object
        Computes all univariate marginals + any multivariate marginals specified in the arguments.
        For instance, to compute the multivariate marginals for (a, b) and (b, c), we'd use:
        sample.marginals(("a", "b"), ("b", "c"))

        Note that these are groupvarnames, not varnames.
        """
        marginals = self._marginal_idxs(joints, computation_strategy=computation_strategy)
        samples = flatten_tree(self.sample)
        samples = {k:v.detach() for (k, v) in samples.items()}
        return Marginals(samples, marginals, self.all_platedims, self.P.varname2groupvarname())

    def _moments_uniform_input(self, moms, computation_strategy=no_checkpoint):
        """
        Must use computation_strategy=NoCheckpoint, as there seems to be a subtle issue in the interaction between
        checkpointing and TorchDims (not sure why it doesn't emerge elsewhere...)
        """
        assert isinstance(moms, list)

        for (varnames, m) in moms:
            if not isinstance(m, RawMoment):
                raise Exception("Moments in sample must be `RawMoment`s (i.e. you must be able to compute them as E[f(x)])")

        flat_sample = flatten_dict(self.sample)
        flat_sample = {k: v.detach() for (k, v) in flat_sample.items()}

        #List of named Js to go into torch.autograd.grad
        J_tensor_list = []
        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
        f_J_torchdim_dict = {}
        #dimension names
        dimss = []

        set_all_platedims = set(self.all_platedims.values())

        for (varnames, m) in moms:
            samples = [flat_sample[varname] for varname in varnames]

            #Check that the variables are heirachically nested within plates.
            platedimss = [set(generic_dims(sample)).intersection(set_all_platedims) for sample in samples]
            longest_platedims = sorted(platedimss, key=len)[-1]
            for platedims in platedimss:
                assert set(platedims).issubset(longest_platedims)

            f = m.f(*samples).detach()
            assert set(generic_dims(f)).intersection(set_all_platedims) == set(longest_platedims)

            dims = tuple(longest_platedims)
            dimss.append(dims)
            dim_sizes = [dim.size for dim in dims]
            sizes = [*dim_sizes, *f.shape]

            J_tensor = t.zeros(sizes, device=self.device, requires_grad=True)
            J_tensor_list.append(J_tensor)
            f_J_torchdim = f*generic_getitem(J_tensor, dims)

            f_J_torchdim_dict[(varnames, m)] = f_J_torchdim
        

        #Compute loss
        L = self._elbo(extra_log_factors=f_J_torchdim_dict, computation_strategy=computation_strategy)

        #marginals as a list
        moments_list = grad(L, J_tensor_list)
        return [generic_getitem(x, dims) for (x, dims) in zip(moments_list, dimss)]

    _moments = torchdim_moments_mixin
    moments = named_moments_mixin

        
def index_into_sample(
        sample: dict, 
        indices: dict[str, Tensor], 
        groupvarname2Kdim:dict[str, Dim], 
        varname2groupvarname:dict[str, str]):
    '''Takes a sample (nested dict of tensors with Kdims) and a dictionary of Kdims to indices.
    Returns a new sample (nested dict of tensors with Ndims instead of Kdims) with the indices
    applied to the sample.'''

    result = {}
    
    for name, value in sample.items():
        assert isinstance(value, (dict, Tensor))

        if isinstance(value, dict):
            result[name] = index_into_sample(value, indices, groupvarname2Kdim, varname2groupvarname)
        elif isinstance(value, Tensor):
            groupvarname = varname2groupvarname[name]
            Kdim = groupvarname2Kdim[groupvarname]

            result[name] = value.detach().order(Kdim)[indices[groupvarname]]

    return result
