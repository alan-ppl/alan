import pytest
import itertools

import torch as t

from alan_simplified import sampling_types
from alan_simplified.utils import generic_dims, generic_order, generic_getitem, generic_all
from alan_simplified.moments import var_from_raw_moment

from model1 import tp as model1
from bernoulli_no_plate import tp as bernoulli_no_plate

tps = [model1, bernoulli_no_plate]
reparams = [True, False]
tp_reparam_sampling_types = list(itertools.product(tps, reparams, sampling_types))


@pytest.mark.parametrize("tp,reparam,sampling_type", tp_reparam_sampling_types)
def test_moments_sample_marginal(tp, reparam, sampling_type):
    """
    tests `marginal.moments` = `sample.moments`
    should be exactly equal
    so we can use small K without incurring large approximation errors.
    """

    sample = tp.problem.sample(K=3, reparam=reparam, sampling_type=sampling_type)
    marginals = sample.marginals()

    sample_moments = sample._moments(tp.moments)
    marginals_moments = marginals._moments(tp.moments)

    for (varname, moment), sm, mm in zip(tp.moments, sample_moments, marginals_moments):
        dims = generic_dims(sm)
        sm = generic_order(sm, dims)
        mm = generic_order(mm, dims)
        assert t.allclose(sm, mm)

@pytest.mark.parametrize("tp,reparam,sampling_type", tp_reparam_sampling_types)
def test_moments_importance_sample(tp, reparam, sampling_type):
    """
    tests `marginal.moments` approx `importance_sample.moments`

    critically, importance_samples draws independent samples from a distribution over K, and
    `sample.moments` or `sample.marginals.moments` computes exact moments wrt this distribution

    Therefore:
    * In the limit as N -> infinity, we expect an exact match.
    * ESS is just N.
    """
    sample = tp.problem.sample(K=tp.moment_K, reparam=reparam, sampling_type=sampling_type)
    marginals = sample.marginals()
    importance_sample = sample.importance_sample(tp.importance_N)

    for varnames, m in tp.moments:
        marginal_moment = marginals._moments(varnames, m)
        is_moment = importance_sample._moments(varnames, m)
        est_var = marginals.moments(varnames, var_from_raw_moment(m))

        stderr = (est_var/tp.importance_N).sqrt() 
        
        assert generic_all(is_moment < marginal_moment + tp.stderrs * stderr)
        assert generic_all(marginal_moment - tp.stderrs * stderr < is_moment)

#def test_moments_ground_truth(tp, sampling_type):
#    """
#    tests `marginal.moments` approx `ground truth`.
#
#    The tp is that we can't easily evaluate the ESS.
#    The obvious approach is to use the ESS for the marginal of the variable of interest (from `sample.marginals`).
#    But that isn't right: the ESS can be reduced because of lack of diversity in other latent variables.
#    Here, we use the minimum ESS across all latent variables in the model.
#    """
#    sample = tp.problem.sample(K=problem.moment_K, reparam=False, sampling_type=sampling_type)
#    marginals = sample.marginals()
#    min_ess = marginals.min_ess()
#
#    for (varnames, m), true_moment in tp.known_moments.items():
#        marginal_moment = marginals.moments(varnames, m)
#        est_var = marginals.moments(varnames, var_from_raw_moment(m))
#
#        stderr = (est_var/min_ess).sqrt() 
#        
#        assert marginal_moment < true_moment + tp.stderrs * stderr
#        assert true_moment - tp.stderrs * stderr < marginal_moment
#
#def test_all(tp, sampling_type):
#    tp.test_moments_sample_marginal(sampling_type)
#    tp.test_moments_importance_sample(sampling_type)
#    tp.test_moments_ground_truth(sampling_type)
#
