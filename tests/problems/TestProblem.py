import torch as t
from alan_simplified.utils import generic_dims, generic_order
from alan_simplified.moments import RawMoment, var_from_raw_moment

class TestProblem():
    def __init__(self, problem, moments, known_moments=None, known_elbo=None, moment_K=30, elbo_K=30, importance_N=1000, stderrs=6):
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
        self.importance_N = importance_N
        self.stderrs = stderrs

    def test_moments_sample_marginal(self, sampling_type):
        """
        tests `sample.moments` = `marginal.moments`
        should be exactly equal
        so we can use small K without incurring large approximation errors.
        """

        sample = self.problem.sample(K=3, reparam=False, sampling_type=sampling_type)
        marginals = sample.marginals()

        sample_moments = sample._moments(self.moments)
        marginals_moments = marginals._moments(self.moments)

        for (varname, moment), sm, mm in zip(self.moments, sample_moments, marginals_moments):
            dims = generic_dims(sm)
            sm = generic_order(sm, dims)
            mm = generic_order(mm, dims)
            assert t.allclose(sm, mm)

    def test_moments_importance_sample(self, sampling_type):
        """
        tests `sample.moments` approx `importance_sample.moments`

        critically, importance_samples draws independent samples from a distribution over K, and
        `sample.moments` or `sample.marginals.moments` computes exact moments wrt this distribution

        Therefore:
        * In the limit as N -> infinity, we expect an exact match.
        * ESS is just N.
        """
        sample = self.problem.sample(K=self.moment_K, reparam=False, sampling_type=sampling_type)
        marginals = sample.marginals()
        importance_sample = sample.importance_sample(self.importance_N)

        for varnames, m in self.moments:
            marginal_moment = marginals._moments(varnames, m)
            is_moment = importance_sample._moments(varnames, m)
            est_var = marginals.moments(varnames, var_from_raw_moment(m))

            stderr = (est_var/self.importance_N).sqrt() 
            
            assert is_moment < marginal_moment + self.stderrs * stderr
            assert marginal_moment - self.stderrs * stderr < is_moment

    def test_moments_ground_truth(self, sampling_type):
        """
        tests `sample.moments` approx `ground truth`.

        The problem is that we can't easily evaluate the ESS.
        The obvious approach is to use the ESS for the marginal of the variable of interest (from `sample.marginals`).
        But that isn't right: the ESS can be reduced because of lack of diversity in other latent variables.
        Here, we use the minimum ESS across all latent variables in the model.
        """
        sample = self.problem.sample(K=self.moment_K, reparam=False, sampling_type=sampling_type)
        marginals = sample.marginals()
        min_ess = marginals.min_ess()

        for (varnames, m), true_moment in self.known_moments.items():
            marginal_moment = marginals.moments(varnames, m)
            est_var = marginals.moments(varnames, var_from_raw_moment(m))

            stderr = (est_var/min_ess).sqrt() 
            
            assert marginal_moment < true_moment + self.stderrs * stderr
            assert true_moment - self.stderrs * stderr < marginal_moment

