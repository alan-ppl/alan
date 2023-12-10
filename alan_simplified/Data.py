from typing import Optional

from functorch.dim import Dim

from .utils import *
from .SamplingType import SamplingType


class Data():
    
    def __init__(self):
        pass

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
        pass