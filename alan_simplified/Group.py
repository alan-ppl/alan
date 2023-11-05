from typing import Optional
from .dist import Dist
from .utils import *
from .SamplingType import SamplingType

class Group(): 
    def __init__(self, **kwargs):
        #Groups can only contain Dist, not Plates/Timeseries/other Groups.
        for varname, dist in kwargs.items():
            if not isinstance(dist, Dist):
                raise Exception("{varname} in a Group should be a Dist, but is actually {type(dist)}")

        if len(kwargs) < 2:
            raise Exception("Groups only make sense if they have two or more random variables, but this group only has {len(kwargs)} random variables")

        self.prog = kwargs
        all_arg_list = [arg for dist in kwargs.values() for arg in dist.all_args]
        self.all_args = list(set(all_arg_list))

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

        scope = {**scope} #This is the modified scope that will be used in the outer plate.
        result = {}       #This is the sample returned.

        Kdim = groupvarname2Kdim[name]
        sample_dims = [Kdim, *active_platedims]

        #resampled scope is the scope used in here when sampling from the Group
        filtered_scope = self.filter_scope(scope)
        resampled_scope = sampling_type.resample_scope(filtered_scope, active_platedims, Kdim)

        for name, dist in self.prog.items():
            tdd = dist.tdd(resampled_scope)
            sample = tdd.sample(reparam, sample_dims, dist.sample_shape)

            scope[name] = sample
            resampled_scope[name] = sample
            result[name] = sample

        return result, scope

    def all_prog_names(self):
        return self.prog.keys()
