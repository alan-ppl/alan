import torch
from functorch.dim import Dim

from .SamplingType import *
from .dist import AlanDist
from .utils import *

def update_scope(scope: dict[str, Tensor], dgpt, name:str, sample):
    """
    Scope is a flat dict mapping variable names to Tensors.

    This function updates the scope after sampling a new thing (either a dist, group or plate).
    """
    if   isinstance(dgpt, AlanDist):
        # sampling from a AlanDist. We get back a tensor and add it to the scope.
        assert isinstance(sample, Tensor)
        scope = {**scope, name: sample}
    elif isinstance(dgpt, Group):
        # sampling from a Group. We get back a flat dict of Tensors, and add all of them to scope.

        #check the sample is a flat dict of tensors.
        assert isinstance(sample, dict)
        for tensor in sample.values():
            assert isinstance(tensor, Tensor)

        scope = {**scope, **sample}
    else:
        #If sampling from a sub-plate, then we don't add those variables to the scope.
        assert isinstance(dgpt, Plate)
        assert isinstance(sample, dict)

    return scope

def update_active_platedims(name, dgpt, active_platedims: list[Dim], all_platedims: dict[str, Dim]):
    """
    active_platedims is from the perspective of the _caller_ plate, not this plate, so we need
    to rearrange things a bit so they make sense from the perspective of this plate,
    """
    if isinstance(dgpt, Plate):
        active_platedims = [all_platedims[name], *active_platedims]
    return active_platedims

class AbstractPlateGroup():
    def __init__(self, **kwargs):
        self.prog = kwargs

    def sample(self, 
               scope:dict[str, Tensor], 
               active_platedims: list[str], 
               all_platedims: dict[str, Dim], 
               sampling_type:SamplingType, 
               Kdim: Dim, 
               reparam):
        """
        Called when sampling from the prior, or when sampling from the approximate posterior.
        We always have a single global Kdim.
        """

        result = {}

        for name, dgpt in self.prog.items():
            sample = dgpt.sample(
                scope=scope,
                active_platedims=update_active_platedims(name, dgpt, active_platedims, all_platedims),
                all_platedims=all_platedims, 
                sampling_type=sampling_type,
                Kdim=Kdim, 
                reparam=reparam
            )
            result[name] = sample

            scope = update_scope(scope, dgpt, name, sample)

        return result

class Plate(AbstractPlateGroup):
    def log_prob(self, samples, scope, active_platedims: list[str], all_platedims: dict[str, Dim], sampling_type):
        """
        Builds a tree of log-probs as a dict, mirroring the structure of a tr.
        
        Follows the scoping rules used by sample.

        Most of the error checking is inside dist.log_prob.
        """
        assert isinstance(samples, dict)

        lps = {}
        for name, dgpt in self.prog.items():
            sample = samples[name]

            lps[name] = dgpt.log_prob(
                sample,
                Kdim,
                scope,
                update_active_platedims(name, dgpt, active_platedims, all_platedims),
                all_platedims, 
                sampling_type
            )

            scope = update_scope(scope, dgpt, name, sample)

        return PlateLPs(active_platedims, K_dims, lps)

    def varname2Kdim(self, K, platename):
        """
        Returns a dict mapping variable names to their Kdims, taking into account groups.
        """
        del platename #Don't need platename.

        result = {}
        for name, dpt in self.prog.items():
            if isinstance(dpt, AlanDist):
                result[name] = Dim(vargroupname2Kname(name), K)
            elif isinstance(dpt, (Plate, Group)):
                result = {**result, **dpt.varname2Kdim(dpt, K, name)}
            else:
                error()
        return result

def vargroupname2Kname(name):
    return f"K_{name}"

class Group(AbstractPlateGroup):
    def __init__(self, **kwargs):
        #Groups can only contain variables, not Plates/Timeseries/other Groups.
        for dist in kwargs.values():
            assert isinstance(dist, AlanDist)

        self.prog = kwargs

    def update_active_platedims(name, active_platedims, all_platedims):
        """
        Groups have the same platedims as the calling Plate.
        """
        return active_platedims

    def varname2Kdim(self, K:int, groupname):
        Kdim = Dim(vargroupname2Kname(groupname), K)
        return {varname: Kdim for varname in self.prog}

def convert_sample_global2local_K(sample, Kdim: Dim, varname2Kdim: dict[str, Dim]):
    """
    Converts a sample (represented as a nested dict) with local Kdim to global Kdim.
    """
    result = {}
    for (k, v) in sample.items():
        if isinstance(v, Tensor):
            result[k] = v.order(Kdim)[varname2Kdim[k]]
        else:
            assert isinstance(v, dict)
            result[k] = convert_sample_global2local_K(v, Kdim, varname2Kdim)
    return result

