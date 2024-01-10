from .Plate import Plate, update_scope
from .utils import *
from .moments import RawMoment, torchdim_moments_mixin, named_moments_mixin

from .Data import Data
from .dist import Dist
from .Plate import Plate
from .Timeseries import Timeseries

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

    def logpq(self, reparam):
        """
        Returns a K-long vector of probabilities
        """
        sample = self.reparam_sample if reparam else self.detached_sample

        return non_mp_log_prob(
            name = None, 
            P = self.problem.P.plate,
            Q = self.problem.Q.plate,
            sample = sample,
            inputs_params=self.problem.inputs_params(),
            data=self.problem.data,
            scope={},
            active_platedims = [],
            all_platedims=self.problem.all_platedims,
            Kdim = self.Kdim,
        )


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
        name,
        P,
        Q,
        sample,
        inputs_params: dict,
        data: dict,
        scope: dict,
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        Kdim: Dim,
        ):
    """
    Iterates through flat.
    """
    if name is not None:
        new_platedim = all_platedims[name]
        active_platedims = [*active_platedims, new_platedim]

    scope = update_scope(scope, inputs_params)
    scope = update_scope(scope, sample)

    set_expected_dims = set([*active_platedims, Kdim])

    lpqs = []
    for k, distQ in Q.flat_prog.items():
        distP = P.flat_prog[k]
        assert not isinstance(distP, Timeseries)
        if isinstance(distQ, Plate):
            assert isinstance(distQ, Plate)
            lpq = non_mp_log_prob(
                name = k,
                P = distP, 
                Q = distQ, 
                sample = sample[k],
                inputs_params = inputs_params[k],
                data = data[k],
                scope = scope,
                active_platedims = active_platedims,
                all_platedims = all_platedims,
                Kdim = Kdim
            )
            assert set(generic_dims(lpq)) == set([Kdim])
        elif isinstance(distQ, Data):
            assert isinstance(distP, Dist)
            assert k in data
            assert k not in sample

            lpq, _ = distP.log_prob(data[k], scope=scope, T_dim=None, K_dim=Kdim)
            assert set(generic_dims(lpq)) == set_expected_dims
            lpq = sum_dims(lpq, active_platedims)
        else:
            assert isinstance(distQ, Dist)
            assert k in sample
            assert k not in data

            lp, _ = distP.log_prob(sample[k], scope=scope, T_dim=None, K_dim=Kdim)
            lq, _ = distQ.log_prob(sample[k], scope=scope, T_dim=None, K_dim=Kdim)
            assert set(generic_dims(lp)) == set_expected_dims
            assert set(generic_dims(lq)) == set_expected_dims

            lp = sum_dims(lp, active_platedims)
            lq = sum_dims(lq, active_platedims)

            lpq = lp-lq

        lpqs.append(lpq)

    sum_lpqs = sum(lpqs)
    assert set(generic_dims(sum_lpqs)) == set([Kdim])
    assert 0 == sum_lpqs.ndim
    return sum_lpqs
