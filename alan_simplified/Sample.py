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
    
#    def marginals(self):
#        #This is an ordered dict.
#        samples = flatten_tree(self.sample) 
#
#        #List of named Js to go into torch.autograd.grad
#        Js_named_list = []
#        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
#        Js_torchdim_dict = {}
#        #dimension names
#        dimnamess = []
#        for (varname, sample) in samples.items():
#            dims = generic_dims(sample)
#            shape = [dim.size for dim in dims]
#            dimnames = [str(dim) for dim in dims]
#            dimnamess.append(dimnames)
#            J_named = t.zeros(*shape, device=sample.device, requires_grad=True, names=dimnames)
#            Js_named_list.append(J_named)
#            J_torchdim = J_named.rename(None)[dims]
#            #Marginals need different names from variables.
#            #This is really a problem in how we're representing trees...
#            Js_torchdim_dict[f"{varname}_marginal"] = J_torchdim
#        Js_torchdim_tree = tensordict2tree(self.P, Js_torchdim_dict)
#
#        #Compute loss
#        L = self.elbo(extra_log_factors=Js_torchdim_tree)
#        #marginals as a list
#        marginals_list = grad(L, Js_named_list)
#
#        #marginals as a flat dict
#        marginals_dict = {}
#        for varname, marginal, dimnames in zip(samples.keys(), marginals_list, dimnamess):
#            marginals_dict[varname] = marginal.refine_names(*dimnames)
#
#        return marginals_dict

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

        return dictdim2named_tensordict(flatten_dict(self.index_in(post_idxs, N_dim)))
    
    def marginals(self):

        #List of named Js to go into torch.autograd.grad
        Js_named_list = []
        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
        Js_torchdim_dict = {}
        #dimension names
        dimnamess = []

        #flatten plates but leave groups nested
        flat_sample = {}
        for name, sample in self.sample.items():
            if isinstance(sample, dict) and name not in self.groupvarname2Kdim:
                flat_sample = {**flat_sample, **flatten_dict(sample)}
            else:
                flat_sample[name] = sample

        
        for groupname in self.groupvarname2Kdim.keys():
            if isinstance(flat_sample[groupname], dict):
                active_platedims = []
                for sample in flat_sample[groupname].values():
                    for name in sample.dims:
                        if str(name) in self.all_platedims:
                            active_platedims.append(self.all_platedims[str(name)])
            else:
                active_platedims = [self.all_platedims[str(name)] for name in flat_sample[groupname].dims if str(name) in self.all_platedims]
                        
            Kdim = self.groupvarname2Kdim[groupname]
            dims = [*active_platedims, Kdim]

            shape = [dim.size for dim in dims]
            dimnames = [*[str(dim) for dim in active_platedims], 'K']
            dimnamess.append(dimnames)
            J_named = t.zeros(shape, requires_grad=True)
            Js_named_list.append(J_named)
            J_torchdim = J_named.rename(None)[dims]
            #Marginals need different names from variables.
            #This is really a problem in how we're representing trees...
            Js_torchdim_dict[f"{groupname}_marginal"] = J_torchdim
        Js_torchdim_tree = tensordict2tree(self.P.plate, Js_torchdim_dict)

        #Compute loss
        L = self.elbo(extra_log_factors=Js_torchdim_tree)
        #marginals as a list
        marginals_list = grad(L, Js_named_list)

        #marginals as a flat dict
        marginals_dict = {}
        for varname, marginal, dimnames in zip(flat_sample.keys(), marginals_list, dimnamess):
            marginals_dict[varname] = marginal.refine_names(*dimnames)

        return marginals_dict

    # def conditionals(self):
    #     """
    #     Returns torchdim tensors because these will only be used internally.
    #     """

    #     #List of named Js to go into torch.autograd.grad
    #     Js_named_list = []
    #     #Flat dict of torchdim tensors to go into elbo as extra_log_factors
    #     Js_torchdim_dict = {}

    #     groupvarname2parents = self.problem.groupvarname2parent_groupvarnames()
    #     groupvarname2active_platedimnames = self.Q.groupvarname2active_platedimnames()
    #     assert set(groupvarname2parents.keys()) == set(groupvarname2active_platedimnames.keys())
    #     groupvarnames = list(groupvarname2active_platedimnames.keys())

    #     dimss = []

    #     for (varname, active_platedimnames) in groupvarname2active_platedimnames.items():
    #         active_platedims = [self.all_platedims[name] for name in active_platedimnames]
    #         Kdimnames = [varname, *groupvarname2parents[varname]]
    #         Kdims = [self.groupvarname2Kdim[Kdimname] for Kdimname in Kdimnames]
    #         dims = [*active_platedims, *Kdims]
    #         dimss.append(dims)

    #         shape = [dim.size for dim in dims]

    #         J_named = t.zeros(shape, requires_grad=True)
    #         Js_named_list.append(J_named)
    #         J_torchdim = J_named.rename(None)[dims]
    #         #Marginals need different names from variables.
    #         #This is really a problem in how we're representing trees...
    #         Js_torchdim_dict[f"{varname}_conditional"] = J_torchdim
    #     Js_torchdim_tree = tensordict2tree(self.P.plate, Js_torchdim_dict)

    #     #Compute loss
    #     L = self.elbo(extra_log_factors=Js_torchdim_tree)
    #     #conditionals as a list
    #     conditionals_list = grad(L, Js_named_list)

    #     #conditionals as a flat dict
    #     conditionals_dict = {}
    #     for varname, marginal, dims in zip(groupvarnames, conditionals_list, dimss):
    #         conditionals_dict[varname] = generic_getitem(marginal, dims)

    #     return conditionals_dict


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
                dimnames = [*[str(dim) for dim in active_platedims]]
                dimnamess.append(dimnames)
                J_named = t.zeros(shape, requires_grad=True)
                Js_named_list.append(J_named)
                if len(dims) > 0:
                    J_torchdim = J_named.rename(None)[dims]
                else:
                    J_torchdim = J_named.rename(None)
                #Moments need different names from variables.
                #This is really a problem in how we're representing trees...
                Js_torchdim_dict[f"{varname}_{f.__name__}"] = J_torchdim * m
                
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