from .Plate import Plate, update_scope
from .utils import *
from .moments import RawMoment, torchdim_moments_mixin, named_moments_mixin

from .Data import Data
from .dist import Dist
from .Plate import Plate, tensordict2tree, flatten_tree
from .Timeseries import Timeseries
from .Split import no_checkpoint
from .ImportanceSample import ImportanceSample
from .Sample import index_into_sample

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

    def logpq(self, sample):
        """
        Returns a K-long vector of probabilities
        """

        result = non_mp_log_prob(
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

        assert result.ndim==0
        assert result.dims==(self.Kdim,)
        return result

    def _elbo(self, sample):
        return self.logpq(sample).order(self.Kdim).logsumexp(0) - math.log(self.Kdim.size)

    def elbo_vi(self, computation_strategy=checkpoint):
        if not self.reparam==True:
            raise Exception("To compute the ELBO with the right gradients for VI you must construct a reparameterised sample using `problem.sample(K, reparam=True)`")
        return self._elbo(self.reparam_sample)

    def elbo_rws(self):
        return self._elbo(self.detached_sample)

    def elbo_nograd(self):
        with t.no_grad():
            result = self._elbo(self.detached_sample)
        return result
    
    def _importance_sample_idxs(self, N: int):
        N_dim = Dim('N', N)

        lps = self.logpq(self.detached_sample)
        lps_max = lps.amax(self.Kdim)

        indices = t.multinomial(t.exp(lps - lps_max), N, replacement=True)

        indices = indices.order(self.Kdim)[N_dim]

        return indices, N_dim

    def importance_sample(self, N:int):
        indices, N_dim = self._importance_sample_idxs(N)

        importance_samples = index_into_non_mp_sample(self.detached_sample, indices, self.Kdim)

        return ImportanceSample(self.problem, importance_samples, N_dim)

    def _moments_uniform_input(self, moms):
        """
        Must use computation_strategy=NoCheckpoint, as there seems to be a subtle issue in the interaction between
        checkpointing and TorchDims (not sure why it doesn't emerge elsewhere...)
        """
        assert isinstance(moms, list)
        lpq = self.logpq(self.detached_sample)
        weights = (lpq - lpq.logsumexp(self.Kdim)).exp()

        flat_sample = flatten_dict(self.detached_sample)

        result = []
        for (varnames, m) in moms:
            args = tuple(flat_sample[varname] for varname in varnames)
            result.append(m.from_marginals(args, weights, self.problem.all_platedims))

        return result

    _moments = torchdim_moments_mixin
    moments = named_moments_mixin

    def update_qem_params(self, lr:float):
        """
        """
        self.problem.P._update_qem_params(lr, self, computation_strategy=no_checkpoint)
        self.problem.Q._update_qem_params(lr, self, computation_strategy=no_checkpoint)


def unify_dims(sample, Kdim, set_all_platedims):
    result = {}
    for k, v in sample.items():
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

def index_into_non_mp_sample(sample, indices, Kdim):
    result = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            result[k] = index_into_non_mp_sample(v, indices, Kdim)
        else:
            result[k] = v.order(Kdim)[indices]
    return result