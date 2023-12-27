import torch as t

from typing import Optional
from .dist import Dist
from .utils import *
from .Sampler import Sampler

class Group(): 
    def __init__(self, **kwargs):
        #Groups can only contain Dist, not Plates/Timeseries/Data/other Groups.
        for varname, dist in kwargs.items():
            if not isinstance(dist, Dist):
                raise Exception("{varname} in a Group should be a Dist, but is actually {type(dist)}")

        if len(kwargs) < 2:
            raise Exception("Groups only make sense if they have two or more random variables, but this group only has {len(kwargs)} random variables")

        self.prog = kwargs
        set_all_arg_list = set([arg for dist in kwargs.values() for arg in dist.all_args])
        self.all_args = set_all_arg_list.difference(kwargs.keys()) #remove dependencies on other variables in the group.

    def filter_scope(self, scope: dict[str, Tensor]):
        return {k: v for (k,v) in scope.items() if k in self.all_args}

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            inputs_params: dict,
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampler:Sampler,
            reparam:bool,
            ):

        result = {}       #This is the sample returned.

        Kdim = groupvarname2Kdim[name]
        sample_dims = [Kdim, *active_platedims]

        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)
        scope = sampler.resample_scope(scope, active_platedims, Kdim)

        for name, dist in self.prog.items():
            tdd = dist.tdd(scope)
            sample = tdd.sample(reparam, sample_dims, dist.sample_shape)

            scope[name]  = sample
            result[name] = sample

        return result
    
    def sample_extended(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:dict):
        
        result = {}       #This is the sample returned.

        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)

        # Loop through all dists in the group and sample from them (plus potentially get 
        # logprobs of original and extended data IF extended_data is provided, i.e. not None)
        for name, dist in self.prog.items():

            childsample = dist.sample_extended(
                sample=sample.get(name),
                name=name,
                scope=scope,
                inputs_params=inputs_params,
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                active_extended_platedims=active_extended_platedims,
                Ndim=Ndim,
                reparam=reparam,
                original_data=original_data,
            )

            scope[name]  = childsample
            result[name] = childsample

        return result
    
    def predictive_ll(
        self,
        sample:dict,
        name:str,
        scope:dict[str, Tensor],
        inputs_params:dict,
        original_platedims:dict[str, Dim],
        extended_platedims:dict[str, Dim],
        original_data: dict[str, Tensor],
        extended_data: dict[str, Tensor]):
        
        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)

        original_lls, extended_lls = {}, {}

        # Loop through all dists in the group and sample from them (plus potentially get 
        # logprobs of original and extended data IF extended_data is provided, i.e. not None)
        for name, dist in self.prog.items():

            child_original_lls, child_extended_lls = dist.predictive_ll(
                sample=sample.get(name),
                name=name,
                scope=scope,
                inputs_params=inputs_params,
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                original_data=original_data,
                extended_data=extended_data
            )

            scope[name]  = sample.get(name)

            original_lls = {**original_lls, **child_original_lls}
            extended_lls = {**extended_lls, **child_extended_lls}

        return original_lls, extended_lls


    def all_prog_names(self):
        return self.prog.keys()
