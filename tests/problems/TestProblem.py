class TestProblem():
    def __init__(self, problem, known_moments=None, known_elbo=None, moment_K=30, elbo_K):
        self.problem = problem
        self.known_moments = known_moments
        self.known_elbo = known_elbo
        self.moment_K = moment_K
        self.elbo_K = elbo_K

    def test_moments_sample_marginal(self, sampling_type):
        """
        tests `sample.moments` = `marginal.moments`
        should be exactly equal
        so we can use small K without incurring large approximation errors.
        """

        sample = self.problem.sample(K=3, reparam=False, sampling_type)



    def test_moments_importance_sample(self, sampling_type):
        """
        tests `sample.moments` approx `importance_sample.moments`

        critically, importance_samples draws independent samples from a distribution over K, and
        `sample.moments` or `sample.marginals.moments` computes exact moments wrt this distribution

        Therefore:
        * In the limit as N -> infinity, we expect an exact match.
        * ESS is just N.
        """
        pass

    def test_moments_ground_truth(self, sampling_type):
        """
        tests `sample.moments` approx `ground truth`.

        The problem is that we can't easily evaluate the ESS.
        The obvious approach is to use the ESS for the marginal of the variable of interest.
        But that isn't quite right: the ESS can be reduced because of lack of diversity in other latent variables.
        Could use minimum of ESS for all latents?

        The other approach is to estimate the moments 10 times, and look at the variance of the overall estimates.


        """
