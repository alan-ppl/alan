class TestProblem():
    def __init__(self, problem, known_moments=None, known_elbo=None, moment_K=30, elbo_K):
        self.problem = problem
        self.known_moments = known_moments
        self.known_elbo = known_elbo
        self.moment_K = moment_K
        self.elbo_K = elbo_K

    def test_moments_sample_marginal(self, sampling_type):
        sample = self.problem.

