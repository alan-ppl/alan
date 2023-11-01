import torch
from functorch.dim import Dim

from .SamplingType import *
from .dist import AlanDist
from .utils import *
from .GroupSample import GroupSample

def update_scope(scope: dict[str, Tensor], name:str, sample):
    """
    Scope is a flat dict mapping variable names to Tensors.

    sample could be a tensor or a dict.
    """
    if isinstance(sample, dict):
        return {**scope, **sample}
    elif isinstance(sample, GroupSample):
        return {**scope, **sample.sample}
    else:
        assert isinstance(sample, Tensor)
        return {name: sample, **scope}

def update_active_platedims(name, dgpt, active_platedims: list[Dim], all_platedims: dict[str, Dim]):
    """
    active_platedims is from the perspective of the _caller_ plate, not this plate, so we need
    to rearrange things a bit so they make sense from the perspective of this plate,
    """
    if isinstance(dgpt, Plate):
        active_platedims = [all_platedims[name], *active_platedims]
    return active_platedims

class PlateTimeseriesGroup():
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

            scope = update_scope(scope, name, sample)

        return result

class PlateTimeseries(PlateTimeseriesGroup):
    pass

class Plate(PlateTimeseries):
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

            scope = update_scope(scope, name, sample)

        return PlateLPs(active_platedims, K_dims, lps)

def vargroupname2Kname(name):
    return f"K_{name}"

class Group(PlateTimeseriesGroup):
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
