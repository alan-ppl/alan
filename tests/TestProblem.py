import torch as t
from alan import no_checkpoint
from alan.utils import generic_dims, generic_order
from alan.moments import RawMoment, var_from_raw_moment

class TestProblem():
    def __init__(self, problem, moments, known_moments=None, known_elbo=None, moment_K=30, elbo_K=30, elbo_iters=20, elbo_gap_cat=1, elbo_gap_perm=1, importance_N=1000, computation_strategy=no_checkpoint):
        """
        `moments` is a list of tuples [("a", Mean), (("a", "b"), Cov)] as expected by e.g. `sample.moments`.
        Currently restricted to raw moments.
        `known_moments` is a dict, {("a", Mean): true_moment}.
        """
        self.problem = problem
        self.moments = moments

        for _, m in moments:
            assert isinstance(m, RawMoment)

        if known_moments is None:
            known_moments = {}
        self.known_moments = known_moments
        self.known_elbo = known_elbo
        self.moment_K = moment_K
        self.elbo_K = elbo_K
        self.elbo_iters = elbo_iters
        self.elbo_gap_cat  = elbo_gap_cat
        self.elbo_gap_perm = elbo_gap_perm
        self.importance_N = importance_N
        self.computation_strategy = computation_strategy
