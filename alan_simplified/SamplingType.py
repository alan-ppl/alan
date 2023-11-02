from .utils import *

class SamplingType:
    """
    In non-factorised approximate posteriors, there are several different approaches to which 
    particles of the parent latent variables to condition the approximate posterior on.
    
    In particular:

    SingleSample (there aren't any particles).
    Parallel (draw K samples from the full joint).
    MixturePermutation (permute the parent particles).
    MixtureCategorical (sample the parents from a uniform Categorical).

    Thus, these classes modify sampling and computing for the approximate posterior.
    In particular, these classes implement:

    resample_scope: This modifies sampling of the approximate posterior, at which point there is just 
    a single global K-dimension.  Modifies the parent latent variables by e.g. permuting or 
    resampling them from a Categorical along the single global K-dimension.  At a high-level, 
    takes a dictionary, `scope`, containing all the parent variables in-scope, and returns a 
    permuted/resampled dictionary.

    reduce_log_prob: This modifies computing the log-probability for the approximate posterior.  At this
    point, we have a different K-dimension for each latent variable/group. Thus, the "raw" lp 
    (taken as input) has a K-dimensions for the current and all parent latent variables.  But we 
    need a log prob with just var_Kdim.  So this method e.g. averages over combinations
    of parent particles (as appropriate) and returns a lp with just a var_Kdim, and no parent
    K-dimensions.
    """
    pass

class SingleSample(SamplingType):
    """
    Draw a single sample, with no K-dimension.

    As there are no K-dimensions, there's no need to modify the scope, or the log_probs.
    """
    @staticmethod
    def resample_scope(scope: dict[str, Tensor], Kdim: None):
        assert Kdim is None
        return scope

    @staticmethod
    def reduce_log_prob(lp: Tensor, name: str, varname2Kdim: dict[str, Dim], active_platedims: list[Dim]):
        return lp


class MultipleSamples(SamplingType):
    pass

class Parallel(MultipleSamples):
    """
    Draw K independent samples from the full joint.
    """
    @staticmethod
    def resample_scope(scope: dict[str, Tensor], Kdim: Dim):
        """
        Doesn't permute/resample previous variables.
        scope: dict of all previously sampled variables in scope.
        """
        return scope

    @staticmethod
    def reduce_log_prob(lp: Tensor, name: str, varname2Kdim: dict[str, Dim], active_platedims: list[Dim]):
        """
        lp: log_prob tensor [*active_platedims, *parent_Kdims, var_Kdim]
        Here, we take the "diagonal" of the parent_Kdims
        returns log_prob tensor with [*active_platedims, var_Kdim]
        """
        #Check that every dim in lp is either in active_platedims, or is a Kdim.
        all_dims = set([*varname2Kdim.values(), *active_platedims])
        lp_dims = generic_dims(lp)
        for dim in lp_dims:
            assert dim in all_dims

        var_Kdim = varname2Kdim[name]
        parent_Kdims = set(lp_dims).difference([var_Kdim, *active_platedims])
        
        #Continue...

class Mixture(MultipleSamples):
    """
    A mixture proposal over all combinations of all particles of parent latent variables.

    Note that while there is one log_prob, there are actually a few different approaches to
    sampling.
    """
    @staticmethod
    def reduce_log_prob(lp: Tensor, name: str, varname2Kdim: dict[str, Dim], active_platedims: list[Dim]):
        """
        lp: log_prob tensor [*active_platedims, *parent_Kdim, var_Kdim]
        Here, we take the "average" over parent_Kdim
        returns log_prob tensor with [*active_platedims, Kdim]
        """
        #Check that every dim in lp is either in active_platedims, or is a Kdim.
        all_dims = set([*varname2Kdim.values(), *active_platedims])
        lp_dims = generic_dims(lp)
        for dim in lp_dims:
            assert dim in all_dims

        var_Kdim = varname2Kdim[name]
        parent_Kdims = set(lp_dims).difference([var_Kdim, *active_platedims])

        #Continue...


class MixturePermutation(Mixture):
    """
    A mixture proposal, where we permute the particles on all the parents.
    """
    @staticmethod
    def resample_scope(scope: dict[str, Tensor], Kdim: Dim):
        """
        This is called as we sample Q, and permutes the particles on the parents
        As such, there is only a single K-dimension.
        
        scope: dict of all previously sampled variables in scope.
        """
        new_scope = {}
        for k,v in scope.items():
            ordered_tensor = v.order(Kdim)
            tdd = TorchDimDist(td.uniform.Uniform, low=0, high=1)
            perm = tdd.sample(False, sample_dims=[Kdim], sample_shape=[]).argsort(Kdim)
            permuted_tensor = ordered_tensor[perm,...]
            new_scope[k] = permuted_tensor

        return new_scope
    
class MixtureCategorical(Mixture):
    """
    A mixture proposal, where we resample the particles on the parents using a uniform Categorical.
    """
    @staticmethod
    def resample_scope(scope: dict[str, Tensor], Kdim: Dim):
        """
        This is called as we sample Q, and permutes the particles on the parents
        As such, there is only a single K-dimension.

        scope: dict of all previously sampled variables in scope.
        """
        new_scope = {}
        for k,v in scope.items():
            ordered_tensor = v.order(Kdim)
            tdd = TorchDimDist(td.categorical.Categorical, probs=t.ones(Kdim.size)/Kdim.size)
            perm = tdd.sample(False, sample_dims=[Kdim], sample_shape=[]).argsort(Kdim)
            permuted_tensor = ordered_tensor[perm,...]
            new_scope[k] = permuted_tensor
            
        return new_scope
