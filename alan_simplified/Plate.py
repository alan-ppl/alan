from typing import Optional

from functorch.dim import Dim

from .utils import *
from .SamplingType import SamplingType
from .dist import Dist
from .Group import Group


class PlateTimeseries():
    def __init__(self, **kwargs):
        self.prog = kwargs

class Plate(PlateTimeseries):

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampling_type:SamplingType,
            reparam:bool):

        if name is not None:
            active_platedims = [*active_platedims, all_platedims[name]]

        parent_scope = scope
        scope = {**scope}
        sample = {}

        for childname, childP in self.prog.items():

            childsample, scope = childP.sample(
                name=childname,
                scope=scope, 
                active_platedims=active_platedims,
                all_platedims=all_platedims,
                groupvarname2Kdim=groupvarname2Kdim,
                sampling_type=sampling_type,
                reparam=reparam,
            )

            sample[childname] = childsample

        return sample, parent_scope

    def groupvarname2Kdim(self, K):
        result = {}
        for childname, childP in self.prog.items():
            if isinstance(childP, (Dist, Group)):
                result[childname] = Dim(f"K_{childname}", K)
            else:
                assert isinstance(childP, Plate)
                result = {**result, **childP.groupvarname2Kdim(K)}
        return result
                


        

class Timeseries(PlateTimeseries):
    pass

