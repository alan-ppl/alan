import torch as t
from functorch.dim import Dim

from .utils import *

def varname2Kname(varname: str):
    return f"K_{varname}"

def sample2groupvarname2Kdim(sample: dict, global_Kdim: Dim):
    """
    Takes a sample, and returns a dict mapping variable names to the corresponding local Kdim.
    """
    result = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            result = {**sample2groupvarname2Kdim(v, global_Kdim), **result}
        else:
            assert isinstance(v, (Tensor, GroupSample))
            local_Kdim = Dim(varname2Kname(k), global_Kdim.size)
            result = {k: local_Kdim, **result}
    return result

def switch_Kdim(tensor: Tensor, current_Kdim: Dim, new_Kdim: Dim):
    """
    Actually replaces a K-dimension in a tesnor.
    """
    return tensor.order(current_Kdim)[new_Kdim]

def _global2local_Kdims(globalK_sample: dict, global_Kdim: Dim, local_Kdims: dict[str, Dim]):
    localK_sample = {}

    for k, v in globalK_sample.items():
        if isinstance(v, dict):
            localK_sample[k] = _global2local_Kdims(v, global_Kdim, local_Kdims)
        elif isinstance(v, GroupSample):
            localK_sample[k] = GroupSample({gk: switch_Kdim(gv, global_Kdim, local_Kdims[k]) for gk, gv in v.samples.items()})
        else:
            assert isinstance(v, Tensor)
            localK_sample[k] = switch_Kdim(v, global_Kdim, local_Kdims[k])
    return localK_sample

def global2local_Kdims(globalK_sample: dict, global_Kdim: Dim):
    groupvarname2Kdim = sample2groupvarname2Kdim(globalK_sample, global_Kdim)
    localK_sample = _global2local_Kdims(globalK_sample, global_Kdim, groupvarname2Kdim)
    return localK_sample, groupvarname2Kdim
