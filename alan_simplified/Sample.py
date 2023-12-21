import random 
import string

from typing import Optional, Union, List

from torch.autograd import grad
from functorch.dim import Dim

from .SamplingType import SamplingType
from .Split import Split
from .Plate import Plate, tensordict2tree, flatten_tree, empty_tree
from .utils import *
from .logpq import logPQ_plate
from .sample_logpq import logPQ_sample
from .BoundPlate import BoundPlate
from .Marginals import Marginals


class Sample():
    def __init__(
            self,
            problem,
            sample: dict,
            groupvarname2Kdim: dict[str, Dim],
            sampling_type: SamplingType,
            split:Split
        ):
        self.problem = problem
        self.sample = sample
        self.groupvarname2Kdim = groupvarname2Kdim
        self.sampling_type = sampling_type
        self.split = split

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

    def elbo(self, extra_log_factors=None):

        if extra_log_factors is None:
            extra_log_factors = empty_tree(self.P.plate)
        assert isinstance(extra_log_factors, dict)
        #extra_log_factors = named2dim_dict(extra_log_factors, self.all_platedims)
        #extra_log_factors = tensordict2tree(self.P.plate, extra_log_factors)

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
            sampling_type=self.sampling_type,
            split=self.split)

        return lp
    
    def resample(self):
        #map Kdimname -> active_platedims
        #map Kdimname -> parent Kdimnames (using all_args on the dist)
        #create a tree mapping
        pass
    
    def importance_sampled_idxs(self, num_samples=1):
        """
        User-facing method that returns reweighted samples.
        """

        #extra_log_factors doesn't make sense for posterior sampling, but is required for
        #one of the internal methods.
        extra_log_factors = empty_tree(self.P.plate)
        assert isinstance(extra_log_factors, dict)

        N_dim = Dim('N', num_samples)
        
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
            sampling_type=self.sampling_type,
            split=self.split,
            indices={},
            num_samples=num_samples,
            N_dim=N_dim,
        )

        return indices, N_dim

    def importance_samples(self, num_samples=1):
        """
        User-facing method that returns reweighted samples.
        """
        post_idxs, N_dim = self.importance_sampled_idxs(num_samples)

        return dim2named_dict(flatten_dict(self.index_in(post_idxs, N_dim)))

    def _marginal_idxs(self, *joints):
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

            J_tensor = t.zeros(*shape, device=self.device, requires_grad=True)
            J_tensor_list.append(J_tensor)
            J_torchdim = J_tensor[dims]
            
            #Names for extra_log_factors must be unique, and not clash with any in the underlying program
            randstr = ''.join(random.choices(string.ascii_lowercase, k=10))
            J_torchdim_dict[f"{groupvarnames_frozenset}_marginal_{randstr}"] = J_torchdim

        J_torchdim_tree = tensordict2tree(self.P.plate, J_torchdim_dict)

        #Compute loss
        L = self.elbo(extra_log_factors=J_torchdim_tree)
        #marginals as a list
        marginals_list = grad(L, J_tensor_list)

        result = {}
        for gvn, dims, marginals in zip(joints, dimss, marginals_list):
            result[gvn] = marginals[dims]

        return result

    def marginals(self, *joints):
        """
        User-facing method that returns a marginals object
        Computes all univariate marginals + any multivariate marginals specified in the arguments.
        For instance, to compute the multivariate marginals for (a, b) and (b, c), we'd use:
        sample.marginals(("a", "b"), ("b", "c"))

        Note that these are groupvarnames, not varnames.
        """
        marginals = self._marginal_idxs(*joints)
        samples = flatten_tree(self.sample)
        samples = {k:v.detach() for (k, v) in samples.items()}
        return Marginals(samples, marginals, self.all_platedims, self.P.varname2groupvarname())


    def moments(self, latent_to_moment: dict[Dim, List]):
        """latent_to_moment should be a dict mapping from latent varname to a list of functions that takes a sample and returns a moment"""
        
        #List of named Js to go into torch.autograd.grad
        Js_named_list = []
        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
        Js_torchdim_dict = {}
        #dimension names
        dimnamess = []
        
        flat_sample = flatten_dict(self.sample)
        
        for (varname, sample) in flat_sample.items():
            if varname not in latent_to_moment:
                continue
            
            active_platedims = [self.all_platedims[str(name)] for name in sample.dims if str(name) in self.all_platedims]
            dims = [*active_platedims]
            
            ms = [f(sample) for f in latent_to_moment[varname]]

            shape = [dim.size for dim in dims]
            for m,f in zip(ms,latent_to_moment[varname]):
                dimnames = [*[str(dim) for dim in active_platedims], Ellipsis]
                dimnamess.append(dimnames)
                J_named = t.zeros(shape, requires_grad=True)
                Js_named_list.append(J_named)
                dimss = [*dims, Ellipsis]

                J_torchdim = J_named.rename(None)[dimss]


                #Moments need different names from variables.
                #This is really a problem in how we're representing trees...
                Js_torchdim_dict[f"{varname}_{f.__name__}"] = m*J_torchdim 
                
        Js_torchdim_tree = tensordict2tree(self.P.plate, Js_torchdim_dict)
        #Compute loss
        L = self.elbo(extra_log_factors=Js_torchdim_tree)
        #marginals as a list
        moments_list = grad(L, Js_named_list)


        #moments as a flat dict
        moments_dict = {}
        for varname, moment, dimnames in zip(Js_torchdim_dict.keys(), moments_list, dimnamess):
            moments_dict[varname] = moment.refine_names(*dimnames)

        return moments_dict
        
        
    def index_in(self, post_idxs: dict[Dim, Tensor], Ndim: Dim):
        '''Takes a sample (nested dict of tensors with Kdims) and a dictionary of Kdims to indices.
        Returns a new sample (nested dict of tensors with Ndims instead of Kdims) with the indices
        applied to the sample.'''

        return index_into_sample(self.sample, post_idxs, Ndim)
    
    def clone_sample(self, sample: dict):
        '''Takes a sample (nested dict of tensors) and returns a new dict with the same structure
        but with copies of the tensors.'''

        result = {}

        for name, value in sample.items():
            if isinstance(value, dict):
                result[name] = self.clone_sample(value)
            else:
                assert isinstance(value, Tensor)
                result[name] = value.clone()

        return result
    
    def _predictive(self, all_platesizes: dict[str, int], reparam: bool, all_data: dict[str, Tensor], num_samples: int, all_inputs: dict[str, Tensor]):
        '''Samples from the predictive distribution of P and, if given all_data, returns the
        log-likelihood of the original data under the predictive distribution of P.'''
        assert isinstance(self.P, BoundPlate)
        assert isinstance(all_platesizes, dict)

        # self.Ndim = Dim('N', num_samples)
        post_idxs, Ndim = self.importance_sampled_idxs(num_samples=num_samples)
        # post_idxs = self.sample_posterior_indices(num_samples=num_samples)
        
        
        # If all_platesizes is missing some plates from self.all_platedims,
        # add them in without changing their sizes.
        for name, dim in self.all_platedims.items():
            if name not in all_platesizes:
                all_platesizes[name] = dim.size

        # Check that all_platesizes contains no extra plates.
        assert set(all_platesizes.keys()) == set(self.all_platedims.keys())

        # Create the new platedims from the platesizes.
        all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}

        # # Will need to add the extended inputs to the scope
        all_inputs_params = tensordict2tree(self.P.plate, named2dim_dict(all_inputs, all_platedims))

        # We have to work on a copy of the sample so that self.sample's own dimensions 
        # aren't changed.
        indexed_sample = self.index_in(post_idxs, Ndim)

        pred_sample, original_ll, extended_ll = self.P.plate.sample_extended(
            sample=indexed_sample,
            name=None,
            scope={},
            inputs_params=all_inputs_params,
            original_platedims=self.all_platedims,
            extended_platedims=all_platedims,
            active_original_platedims=[],
            active_extended_platedims=[],
            Ndim=Ndim,
            reparam=reparam,
            original_data=self.problem.data,
            extended_data=all_data
        )

        return pred_sample, original_ll, extended_ll, Ndim

    def predictive_sample(self, all_platesizes: dict[str, int], reparam: bool, num_samples=1, all_inputs={}):
        '''Returns a dictionary of samples from the predictive distribution.'''
        pred_sample, _, _, _ = self._predictive(
            all_platesizes=all_platesizes,
            reparam=reparam, 
            all_data=None,
            num_samples=num_samples,
            all_inputs=all_inputs)

        return flatten_dict(pred_sample)

    def predictive_ll(self, all_platesizes: dict[str, int], reparam: bool, all_data: dict[str, Tensor], num_samples=1, all_inputs={}):
        '''This function returns the predictive log-likelihood of the test data (all_data - train_data), given 
        the training samples and the predictive samples.'''        
        _, lls_train, lls_all, Ndim = self._predictive(
            all_platesizes=all_platesizes,
            reparam=reparam, 
            all_data=all_data,
            num_samples=num_samples,
            all_inputs=all_inputs)

        # If we have lls for a variable in the training data, we should also have lls
        # for it in the all (training+test) data.
        assert set(lls_all.keys()) == set(lls_train.keys())

        result = {}
        for varname in lls_all:
            ll_all   = lls_all[varname]
            ll_train = lls_train[varname]

            dims_all   = [dim for dim in ll_all.dims   if dim is not Ndim]
            dims_train = [dim for dim in ll_train.dims if dim is not Ndim]
            assert len(dims_all) == len(dims_train)

            if 0 < len(dims_all):
                # Sum over plates
                ll_all   = ll_all.sum(dims_all)
                ll_train = ll_train.sum(dims_train)

            # Take mean over Ndim
            result[varname] = logmeanexp_dims(ll_all - ll_train, (Ndim,))

        return result
        
def index_into_sample(sample: dict, indices: dict[Dim, Tensor], Ndim: Dim):
    '''Takes a sample (nested dict of tensors with Kdims) and a dictionary of Kdims to indices.
    Returns a new sample (nested dict of tensors with Ndims instead of Kdims) with the indices
    applied to the sample.'''

    result = {}
    
    for name, value in sample.items():
        if isinstance(value, dict):
            result[name] = index_into_sample(value, indices, Ndim)
        elif isinstance(value, Tensor):
            assert isinstance(value, Tensor)
            assert Ndim not in set(generic_dims(value))

            temp = value

            for dim in list(set(generic_dims(value)).intersection(set(indices.keys()))):
                assert Ndim in set(generic_dims(indices[dim]))

                temp = temp.order(dim)[indices[dim]]


            result[name] = temp

    return result
