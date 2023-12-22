from typing import Optional
from .utils import *
from .Plate import flatten_tree, tensordict2tree
from .moments import uniformise_moment_args, postproc_moment_outputs

class AbstractImportanceSample():
    def dump(self):
        """
        User-facing method that returns a flat dict of named tensors, with N as the first dimension.
        """
        return dim2named_dict(self.samples_flatdict)

    def moments(self, *raw_moms):
        moms = uniformise_moment_args(raw_moms)

        result = []
        for varnames, m in moms:
            samples = tuple(self.samples_flatdict[varname] for varname in varnames)
            result.append(m.from_samples(samples, self.Ndim))

        return postproc_moment_outputs(result, raw_moms)

class ImportanceSample(AbstractImportanceSample):
    def __init__(self, problem, samples_tree, Ndim):
        """
        samples is tree-structured torchdim (as we might need to use it for extended).
        """
        self.problem = problem
        self.samples_tree = samples_tree
        self.samples_flatdict = flatten_tree(samples_tree)
        self.Ndim = Ndim

    def extend(self, extended_platesizes:dict[str, int], reparam:bool, extended_inputs: Optional[dict[str, Tensor]]):
        """
        User-facing method that extends sample by drawing the rest from the prior.

        Note: doesn't take any extended data!!!

        returns ExtendedImportanceSample
        """

        assert isinstance(extended_platesizes, dict)
        assert isinstance(reparam, bool)
        assert isinstance(extended_inputs, dict) or extended_inputs is None
        if extended_inputs is None:
            extended_inputs = {}
        
        # If all_platesizes is missing some plates from self.problem.all_platedims,
        # add them in without changing their sizes.
        for name, dim in self.problem.all_platedims.items():
            if name not in extended_platesizes:
                extended_platesizes[name] = dim.size

        # Check that extend_platesizes contains no extra plates.
        assert set(extended_platesizes.keys()) == set(self.problem.all_platedims.keys())

        # Create the new platedims from the platesizes.
        extended_platedims = {name: Dim(name, size) for name, size in extended_platesizes.items()}

        # Will need to add the extended inputs to the scope
        all_inputs_params = tensordict2tree(self.problem.P.plate, named2dim_dict(extended_inputs, extended_platedims))

        extended_sample = self.problem.P.plate.sample_extended(
            sample=self.samples_tree,
            name=None,
            scope={},
            inputs_params=all_inputs_params,
            original_platedims=self.problem.all_platedims,
            extended_platedims=extended_platedims,
            active_extended_platedims=[],
            Ndim=self.Ndim,
            reparam=reparam,
            original_data=self.problem.data,
        )

        return ExtendedImportanceSample(self.problem, extended_sample, self.Ndim, extended_platedims, extended_inputs)

class ExtendedImportanceSample(AbstractImportanceSample):
    def __init__(self, problem, samples_tree, Ndim, extended_platedims, extended_inputs):
        """
        samples is tree-structured torchdim (as we might need to use it for extended).
        """
        self.problem = problem
        self.samples_tree = samples_tree
        self.samples_flatdict = flatten_tree(samples_tree)
        self.Ndim = Ndim
        self.extended_platedims = extended_platedims
        self.extended_inputs = extended_inputs

    def predictive_ll(self, data:dict[str, Tensor]):
        '''
        User-facing method that computes the predictive log-likelihood of the extended data.
        '''
        assert isinstance(data, dict)

        # Convert data to torchdim
        extended_data = named2dim_tensordict(self.extended_platedims, data)
        
        original_data = flatten_tree(self.problem.data)

        # If data is missing (i.e. not being extended), add it in from the original(non-extended) data.
        for name, data_tensor in original_data.items():
            if name not in data.keys():
                data[name] = data_tensor

        # Check that data contains no extra data names.
        assert set(data.keys()) == set(original_data.keys())

        # Will need to add the extended inputs to the scope
        all_inputs_params = tensordict2tree(self.problem.P.plate, named2dim_dict(self.extended_inputs, self.extended_platedims))

        lls_train, lls_all = self.problem.P.plate.predictive_ll(
            sample=self.samples_tree,
            name=None,
            scope={},
            inputs_params=all_inputs_params,
            original_platedims=self.problem.all_platedims,
            extended_platedims=self.extended_platedims,
            original_data=original_data,
            extended_data=extended_data,
        )

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

