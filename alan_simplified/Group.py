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

        self.prog = kwargs
        self.all_args = list(set([dist.all_args for dist in self.kwargs.values()]))

    def filter_scope(self, scope: dict[str, Tensor]):
        return {k: v for (k,v) in scope.items() if k in self.all_args}

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampling_type:SamplingType,
            reparam:bool):

        scope = {**scope}

        Kdim = groupvarname2Kdim[name]
        sample_dims = [Kdim, *active_platedims]

        filtered_scope = self.filter_scope(scope)
        resampled_scope = sampling_type.resample_scope(filtered_scope, active_platedims, Kdim)

        result = {}
        for name, dist in self.prog.items():
            tdd = dist.tdd(resampled_scope)
            sample = tdd.sample(reparam, sample_dims, dist.sample_shape)

            scope[name] = scope
            result[name] = result

        return result, scope
