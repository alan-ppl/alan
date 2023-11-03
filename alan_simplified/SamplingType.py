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
    def resample_scope(scope: dict[str, Tensor], Kdim: None, active_platedims: list[Dim]):
        assert Kdim is None
        return scope

    @staticmethod
    def reduce_log_prob(self, lp: Tensor, Kdim: Dim, all_Kdims: list[Dim], active_platedims: list[Dim]):
        return lp


class MultipleSamples(SamplingType):
    pass

class Parallel(MultipleSamples):
    """
    Draw K independent samples from the full joint.
    """
    @staticmethod
    def resample_scope(self, scope: dict[str, Tensor], Kdim: None, active_platedims: list[Dim]):
        """
        Doesn't permute/resample previous variables.
        scope: dict of all previously sampled variables in scope.
        """
        return scope

    @staticmethod
    def reduce_log_prob(lp: Tensor, Kdim: Dim, all_Kdims: list[Dim], active_platedims: list[Dim]):
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
        
        if len(parent_Kdims) > 0:
            idxs = [t.arange(Kdim.size)[Kdim] for K in parent_Kdims]
            lp = lp.order(*parent_Kdims)[idxs]
        
        return lp

def Kdim2varname2tensors(scope: dict[str, Tensor], Kdim:Dim, active_platedims: list[Dim]):
    """
    Must permute K-dimensions _not_ just tensors.
    But because of groups there could be several tensors with the same K-dimension.
    We therefore put together a dictionary mapping Kdims to all tensors with that K-dimension.

    Everything in scope is either:
      Sample (one  K-dimension)
      Input  (zero K-dimensions)
      Param  (zero K-dimensions)

    Therefore, each tensor should only ever have zero or one K-dimension
    """
    #dict[Dim, dict[str, Tensor]]
    Kdim2varname2tensor = {}
    for varname, tensor in scope.items():
        dims = generic_dims(tensor)
        Kdims = set(dims).difference(active_platedims)
        assert len(Kdims) in [0, 1]
        if len(Kdims) == 1:
            Kdim = Kdims[0]
            if Kdim not in Kdim2tensors:
                Kdim2tensors[Kdim] = {}
            else:
                #If two tensors have the same K-dimension and thus are part of the
                #same group, then they should have exactly the same K/plate dims
                dims_prev = generic_dims(next(iter(Kdim2tensors.values())))
                assert set(generic_dims(tensor)) == set(dims_prev)
            Kdim2tensors[Kdim][varname] = tensor
    return Kdim2tensors

class Mixture(MultipleSamples):
    """
    A mixture proposal over all combinations of all particles of parent latent variables.

    Note that while there is one log_prob, there are actually a few different approaches to
    sampling.
    """
    @staticmethod
    def resample_scope(scope: dict[str, Tensor], Kdim: Dim, active_platedims: list[Dim]):
        """
        This is called as we sample Q, and permutes the particles on the parents

        Kdim is the desired Kdim of the new sample.  That isn't the
        same as the K-dimensions appearing in tensors in scope.  That's because 
        all the variables in scope have their own K-dimensions
        """

        scope = {**scope}
        for var_Kdim,varname2tensor in Kdim2varname2tensors(scope, Kdim, active_platedims).items():
            tensor0 = next(iter(varname2tensor.values()))

            dims = set(generic_dims(tensor0))
            perm = self.perm(dims=dims, Kdim=var_Kdim)

            for varname, tensor in varname2tensor.items():
                new_scope[varname] = tensor.order(var_Kdim)[perm,...][Kdim]

        return new_scope

    @staticmethod
    def reduce_log_prob(lp: Tensor, Kdim: Dim, active_platedims: list[Dim]):
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

        return logmeanexp_dims(lp, dims=parent_Kdims)




class MixturePermutation(Mixture):
    """
    A mixture proposal, where we permute the particles on all the parents.
    """
    @staticmethod
    def perm(self, dims:list[Dim], Kdim:Dim):
        tdd = TorchDimDist(td.uniform.Uniform, low=0, high=1)
        return tdd.sample(False, sample_dims=dims, sample_shape=[]).argsort(Kdim)
    
class MixtureCategorical(Mixture):
    """
    A mixture proposal, where we resample the particles on the parents using a uniform Categorical.
    """
    @staticmethod
    def perm(self, dims:list[Dim], Kdim:Dim):
        tdd = TorchDimDist(td.categorical.Categorical, probs=t.ones(Kdim.size)/Kdim.size)
        platedims = set(dims)
        platedims.remove(Kdim)
        return tdd.sample(False, sample_dims=platedims, sample_shape=[])[Kdim,...]
