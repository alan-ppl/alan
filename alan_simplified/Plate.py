from typing import Optional

from .utils import *
from .SamplingType import SamplingType

class PlateTimeseries():
    def __init__(self, **kwargs):
        self.prog = kwargs


class Plate(PlateTimeseries):

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            active_platedims:list[Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampling_type:SamplingType,
            reparam:bool
            ):

        if name is not None:
            active_platedims = [*active_platedims, all_platedims[name]]

        parent_scope = scope
        scope = {**scope, **inputs_params.values}
        sample = {}

        for childname, childP in self.prog.items():

            childsample, scope = childP.sample(
                name=childname,
                scope=scope, 
                active_platedims=active_platedims,
                groupvarname2Kdim=groupvarname2Kdim,
                sampling_type=sampling_type
            )

            sample[childname] = childsample

        return sample, parent_scope


        

class Timeseries(PlateTimeseries):
    pass

