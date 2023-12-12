from typing import Optional, Union, List
from torch.autograd import grad

from .Plate import Plate, tree_values, update_scope_inputs_params, update_scope_sample, tensordict2tree

from .BoundPlate import BoundPlate
from .Group import Group
from .utils import *
from .reduce_Ks import reduce_Ks, sample_Ks
from .Split import Split
from .Sample import Sample
from .SamplingType import SamplingType
from .dist import Dist
from .logpq import logPQ_plate, logPQ_group, logPQ_dist

PBP = Union[Plate, BoundPlate]


class IndexedSample():
    def __init__(self, sample: Sample, post_idxs: dict[Dim, Tensor]):
        assert isinstance(sample, Sample)
        assert isinstance(post_idxs, dict)

        # Check that we have an index for each Kdim in the sample
        assert set(post_idxs.keys()) == set(sample.groupvarname2Kdim.values())

        # Check that the values (indexing tensors) in post_idxs contain only plate dims (in sample.all_platedims) and one Ndim
        idx_dims = set()
        for idx in post_idxs.values():
            idx_dims = idx_dims.union(set(generic_dims(idx)))

        self.Ndim = list(idx_dims.difference(sample.all_platedims.values()))[0]

        assert idx_dims.difference({self.Ndim}).issubset(set(sample.all_platedims.values()))

        self.original_sample = sample
        self.post_idxs = post_idxs

        self.sample = self.index_in(sample.sample, post_idxs)

    
    def index_in(self, sample: dict, post_idxs: dict[Dim, Tensor]):
        '''Takes a sample (nested dict of tensors with Kdims) and a dictionary of Kdims to indices.
        Returns a new sample (nested dict of tensors with Ndims instead of Kdims) with the indices
        applied to the sample.'''

        result = {}

        for name, value in sample.items():
            if isinstance(value, dict):
                result[name] = self.index_in(value, post_idxs)
            elif isinstance(value, Tensor):
                assert isinstance(value, Tensor)

                temp = value

                for dim in list(set(generic_dims(value)).intersection(set(post_idxs.keys()))):
                    temp = temp.order(dim)[post_idxs[dim]]
 
                result[name] = temp


        return result
    
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
    
    def _predictive(self, P: PBP, all_platesizes: dict[str, int], reparam: bool, all_data: dict[str, Tensor]):
        '''Samples from the predictive distribution of P and, if given all_data, returns the
        log-likelihood of the original data under the predictive distribution of P.'''
        assert isinstance(P, (Plate, BoundPlate))
        assert isinstance(all_platesizes, dict)

        # If all_platesizes is missing some plates from self.sample.all_platedims,
        # add them in without changing their sizes.
        for name, dim in self.original_sample.all_platedims.items():
            if name not in all_platesizes:
                all_platesizes[name] = dim.size

        # Check that all_platesizes contains no extra plates.
        assert set(all_platesizes.keys()) == set(self.original_sample.all_platedims.keys())

        # Create the new platedims from the platesizes.
        all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}

        # We have to work on a copy of the sample so that self.sample's own dimensions 
        # aren't changed.
        sample_copy = self.clone_sample(self.sample)

        pred_sample, original_ll, extended_ll = P.sample_extended(
            sample=sample_copy,
            name=None,
            scope={},
            inputs_params=P.inputs_params(self.original_sample.all_platedims),
            original_platedims=self.original_sample.all_platedims,
            extended_platedims=all_platedims,
            active_original_platedims=[],
            active_extended_platedims=[],
            Ndim=self.Ndim,
            reparam=reparam,
            original_data=self.original_sample.problem.data,
            extended_data=all_data
        )

        return pred_sample, original_ll, extended_ll

    def predictive_sample(self, P: PBP, all_platesizes: dict[str, int], reparam: bool):
        '''Returns a dictionary of samples from the predictive distribution.'''
        pred_sample, _, _ = self._predictive(
            P=P, 
            all_platesizes=all_platesizes,
            reparam=reparam, 
            all_data=None)

        return pred_sample

    def predictive_ll(self, P: PBP, all_platesizes: dict[str, int], reparam: bool, all_data: dict[str, Tensor]):
        '''This function returns the predictive log-likelihood of the test data (all_data - train_data), given 
        the training samples and the predictive samples.'''        
        _, lls_train, lls_all = self._predictive(
            P=P, 
            all_platesizes=all_platesizes,
            reparam=reparam, 
            all_data=all_data)

        # If we have lls for a variable in the training data, we should also have lls
        # for it in the all (training+test) data.
        assert set(lls_all.keys()) == set(lls_train.keys())

        result = {}
        for varname in lls_all:
            ll_all   = lls_all[varname]
            ll_train = lls_train[varname]

            dims_all   = [dim for dim in ll_all.dims   if dim is not self.Ndim]
            dims_train = [dim for dim in ll_train.dims if dim is not self.Ndim]
            assert len(dims_all) == len(dims_train)

            if 0 < len(dims_all):
                # Sum over plates
                ll_all   = ll_all.sum(dims_all)
                ll_train = ll_train.sum(dims_train)

            # Take mean over Ndim
            result[varname] = logmeanexp_dims(ll_all - ll_train, (self.Ndim,))

        return result
    
    
    def moments(self, latent_to_moment: dict[Dim, List], input_sample: dict, post_idxs: dict[Dim, Tensor]):
        """latent_to_moment should be a dict mapping from latent varname to a list of functions that takes a sample and returns a moment"""
        
        #List of named Js to go into torch.autograd.grad
        Js_named_list = []
        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
        Js_torchdim_dict = {}
        #dimension names
        dimnamess = []
        
        groupvarname2active_platedimnames = input_sample.Q.groupvarname2active_platedimnames()
        groupvarnames = list(groupvarname2active_platedimnames.keys())

        indexed_sample = self.index_in(input_sample.sample, post_idxs)
        
        for (varname, active_platedimnames) in groupvarname2active_platedimnames.items():
            if varname not in latent_to_moment:
                continue
            
            
            
            active_platedims = [input_sample.all_platedims[name] for name in active_platedimnames]
            Kdim = input_sample.groupvarname2Kdim[varname]
            dims = [*active_platedims, Kdim]
            
            if active_platedimnames != []:
                for name in active_platedimnames:
                    sample = indexed_sample[name]
                
                ms = [f(sample[varname]) for f in latent_to_moment[varname]]
            else:
                ms = [f(indexed_sample[varname]) for f in latent_to_moment[varname]]
            print(ms)

            shape = [dim.size for dim in dims]
            dimnames = [*[str(dim) for dim in active_platedims], 'K']
            dimnamess.append(dimnames)
            J_named = t.zeros(shape, requires_grad=True)
            Js_named_list.append(J_named)
            J_torchdim = J_named.rename(None)[dims]
            #Moments need different names from variables.
            #This is really a problem in how we're representing trees...
            for m,f in zip(ms,latent_to_moment[varname]):
                Js_torchdim_dict[f"{varname}_{f}"] = J_torchdim * m
                
        Js_torchdim_tree = tensordict2tree(input_sample.P, Js_torchdim_dict)

        #Compute loss
        L = input_sample.elbo(extra_log_factors=Js_torchdim_tree)
        #marginals as a list
        moments_list = grad(L, Js_named_list)

        #moments as a flat dict
        moments_dict = {}
        for varname, moment, dimnames in zip(latent_to_moment.keys(), moments_list, dimnamess):
            moments_dict[varname] = moment.refine_names(*dimnames)

        return moments_dict