from typing import Optional
from .dist import Dist
from .utils import *
from .SamplingType import SamplingType

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
            sampling_type:SamplingType,
            reparam:bool):

        result = {}       #This is the sample returned.

        Kdim = groupvarname2Kdim[name]
        sample_dims = [Kdim, *active_platedims]

        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)
        scope = sampling_type.resample_scope(scope, active_platedims, Kdim)

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
            active_original_platedims:list[Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:Optional[dict[str, Tensor]],
            extended_data:Optional[dict[str, Tensor]]):

        result = {}       #This is the sample returned.

        original_ll = {}
        extended_ll = {}

        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)

        # Loop through all dists in the group and sample from them (plus potentially get 
        # logprobs of original and extended data IF extended_data is provided, i.e. not None)
        for name, dist in self.prog.items():

            childsample, child_original_ll, child_extended_ll = dist.sample_extended(
                sample=sample.get(name),
                name=name,
                scope=scope,
                inputs_params=inputs_params,
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                active_original_platedims=active_original_platedims,
                active_extended_platedims=active_extended_platedims,
                Ndim=Ndim,
                reparam=reparam,
                original_data=original_data,
                extended_data=extended_data
            )

            scope[name]  = childsample
            result[name] = childsample

            original_ll = {**original_ll, **child_original_ll}
            extended_ll = {**extended_ll, **child_extended_ll}

        return result, original_ll, extended_ll

    def all_prog_names(self):
        return self.prog.keys()
