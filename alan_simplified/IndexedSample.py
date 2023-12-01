from typing import Optional, Union

from .Plate import Plate, tree_values, update_scope_inputs_params, update_scope_sample
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
            else:
                assert isinstance(value, Tensor)

                temp = value

                for dim in list(set(generic_dims(value)).intersection(set(post_idxs.keys()))):
                    temp = temp.order(dim)[post_idxs[dim]]
 
                result[name] = temp

        return result
    
    def predictive_sample(self, P: PBP, all_platesizes: dict[str, int], reparam: bool):
        '''Returns a dictionary of samples from the predictive distribution.'''
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

        pred_sample = P.sample_extended(
            sample=self.sample,
            name=None,
            scope={},
            inputs_params=P.inputs_params(self.original_sample.all_platedims),
            original_platedims=self.original_sample.all_platedims,
            extended_platedims=all_platedims,
            active_original_platedims=[],
            active_extended_platedims=[],
            Ndim=self.Ndim,
            reparam=reparam,
            data=self.original_sample.problem.data
        )

        return pred_sample

    def predictive_ll(
        P:Plate,
        train_samples: dict,
        all_samples: dict):
        '''This function returns the predictive log-likelihood, given the training samples and the
        predictive samples.'''

        # calculate and return: ll(all_samples) - ll(train_samples)

        pass