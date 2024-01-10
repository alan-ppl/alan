from .Plate import Plate, tensordict2tree, flatten_tree, empty_tree
from .utils import *
from .moments import RawMoment, torchdim_moments_mixin, named_moments_mixin

class SampleNonMP:
    def __init__(
            self,
            problem,
            sample,
            groupvarname2Kdim,
            reparam,
        ):
        
        self.problem = problem
        self.reparam = reparam

        self.Kdim = Dim('K', next(iter(groupvarname2Kdim.values())).size)
        set_all_platedims = set(self.problem.all_platedims.values())

        sample = unify_dims(sample, self.Kdim, set_all_platedims)

        if self.reparam:
            self.reparam_sample = sample
            self.detached_sample = detach_dict(sample)
        else:
            self.detached_sample = sample

    def logpq(self):
        """
        Returns a K-long vector of probabilities
        """


def unify_dims(d, Kdim, set_all_platedims):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = unify_dims(v, Kdim, set_all_platedims)
        else:
            assert isinstance(v, Tensor)
            v_Kdims = list(set(v.dims).difference(set_all_platedims))
            assert 1==len(v_Kdims)
            result[k] = v.order(v_Kdims[0])[Kdim]
    return result

def non_mp_log_prob(
        P,
        Q,
        sample,
        inputs_params: dict,
        data: dict,
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        ):
    """
    Iterates through flat.
    """
    pass

