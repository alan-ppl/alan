import pytest
import itertools

import torch as t

from alan_simplified import sampling_types, Sample, PermutationSampler, CategoricalSampler, checkpoint, no_checkpoint
from alan_simplified.Marginals import Marginals
from alan_simplified.utils import generic_dims, generic_order, generic_getitem, generic_all, generic_allclose
from alan_simplified.moments import var_from_raw_moment, RawMoment

from model1 import tp as model1
from bernoulli_no_plate import tp as bernoulli_no_plate
from linear_gaussian import tp as linear_gaussian
from linear_gaussian_two_params import tp as linear_gaussian_two_params
from linear_gaussian_two_params_dangling import tp as linear_gaussian_two_params_dangling
from linear_gaussian_two_params_corr_Q import tp as linear_gaussian_two_params_corr_Q
from linear_gaussian_two_params_corr_Q_reversed import tp as linear_gaussian_two_params_corr_Q_reversed
from linear_gaussian_latents import tp as linear_gaussian_latents

tps = [
    model1, 
    bernoulli_no_plate, 
    linear_gaussian, 
    linear_gaussian_two_params,
    linear_gaussian_two_params_dangling,
    #linear_gaussian_two_params_corr_Q,
    #linear_gaussian_two_params_corr_Q_reversed,
    #linear_gaussian_latents,
]#, linear_multivariate_gaussian]
reparams = [True, False]
splits = [checkpoint, no_checkpoint, None]

tp_sampling_types = list(itertools.product(tps, sampling_types))
tp_reparam_sampling_types = list(itertools.product(tps, reparams, sampling_types))
tp_splits = list(itertools.product(tps, splits))

def moment_stderr(marginals, varnames, moment):
    """
    returns a dict mapping (varname, moment) onto a tuple of moment + stderr.
    """
    assert isinstance(marginals, Marginals)
    assert isinstance(moment, RawMoment)

    min_ess = marginals.min_ess()

    result = {}
    marginal_moment = marginals.moments(varnames, moment)
    est_var = marginals.moments(varnames, var_from_raw_moment(moment))

    stderr = (est_var/min_ess).sqrt() 

    return (marginal_moment, stderr)

def combine_stderrs(stderr1, stderr2):
    return (stderr1**2 + stderr2**2).sqrt()


@pytest.mark.parametrize("tp,reparam,sampling_type", tp_reparam_sampling_types)
def test_moments_sample_marginal(tp, reparam, sampling_type):
    """
    tests `marginal.moments` = `sample.moments`
    should be exactly equal
    so we can use small K without incurring large approximation errors.
    """

    sample = tp.problem.sample(K=3, reparam=reparam, sampling_type=sampling_type)
    marginals = sample.marginals()

    for (varnames, moment) in tp.moments:
        sample_moments = sample._moments(varnames, moment)
        marginals_moments = marginals._moments(varnames, moment)

        assert generic_allclose(sample_moments, marginals_moments)

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

@pytest.mark.parametrize("tp,reparam,sampling_type", tp_reparam_sampling_types)
def test_moments_ground_truth(tp, reparam, sampling_type):
    """
    tests `marginal.moments` approx `ground truth`.

    The tp is that we can't easily evaluate the ESS.
    The obvious approach is to use the ESS for the marginal of the variable of interest (from `sample.marginals`).
    But that isn't right: the ESS can be reduced because of lack of diversity in other latent variables.
    Here, we use the minimum ESS across all latent variables in the model.
    """
    sample = tp.problem.sample(K=tp.moment_K, reparam=False, sampling_type=sampling_type)
    marginals = sample.marginals()


    for (varnames, m), true_moment in tp.known_moments.items():
        marginal_moment, stderr = moment_stderr(marginals, varnames, m)
        
        assert generic_all(true_moment < marginal_moment + 6*stderr)
        assert generic_all(marginal_moment - 6*stderr < true_moment)

@pytest.mark.parametrize("tp,sampling_type", tp_sampling_types)
def test_elbo_ground_truth(tp, sampling_type):
    """
    tests `sample.elbo` against ground truth
    """
    if tp.known_elbo is not None:
        N_elbos = tp.elbo_iters
        elbos = []
        for _ in range(N_elbos):
            elbos.append(tp.problem.sample(K=tp.elbo_K, reparam=False, sampling_type=sampling_type).elbo_nograd())
        elbo_tensor = t.stack(elbos)

        sample_mean = elbo_tensor.mean()
        sample_var  = elbo_tensor.var()

        print((sample_mean, sample_var))
        
        stderr_in_mean = (sample_var/N_elbos).sqrt()

        max_mean = sample_mean + 6*stderr_in_mean
        min_mean = sample_mean - 6*stderr_in_mean

        #https://math.stackexchange.com/questions/72975/variance-of-sample-variance
        stderr_in_var = (2 * sample_var**2/N_elbos).sqrt()

        max_var = sample_var + 6*stderr_in_var

        #Assumes ELBO is Gaussian; we know model evidence = E[exp(elbo)]
        #In that case, exp(elbo) is log-normal.
        #We know that E[exp(elbo)] = exp(mean + var/2)
        #where mean and var are the true mean and variance of the distribution over elbos.
        #but from a finite sample, we don't know the true mean and variance; instead we 
        #can get a good estimate of the max possible mean and var.
        max_elbo = max_mean + max_var/2
        #The expected ELBO is a lower-bound on the true model evidence
        min_elbo = min_mean

        print((min_elbo, tp.known_elbo, max_elbo))

        assert            tp.known_elbo < max_elbo
        assert min_elbo < tp.known_elbo

        elbo_gap = tp.elbo_gap_cat if sampling_type is CategoricalSampler else tp.elbo_gap_perm
        assert max_elbo - min_elbo < elbo_gap

@pytest.mark.parametrize("tp,reparam,sampling_type", tp_reparam_sampling_types)
def test_moments_vs_moments(tp, reparam, sampling_type):
    """
    tests `marginal.moments` against each other for different reparam and sampling_type.
    """
    base_marginals = tp.problem.sample(K=tp.moment_K, reparam=False, sampling_type=PermutationSampler).marginals()
    test_marginals = tp.problem.sample(K=tp.moment_K, reparam=reparam, sampling_type=sampling_type).marginals()

    for (varnames, moment) in tp.moments:
        base_moment, base_stderr = moment_stderr(base_marginals, varnames, moment)
        test_moment, test_stderr = moment_stderr(test_marginals, varnames, moment)

        diff = base_moment - test_moment
        stderr = combine_stderrs(base_stderr, test_stderr)

        assert generic_all(                diff < tp.stderrs * stderr)
        assert generic_all(-tp.stderrs * stderr < diff)

@pytest.mark.parametrize("tp,split", tp_splits)
def test_split_elbo_vi(tp, split):
    """
    tests `sample.elbo_vi` against each other for different splits
    """
    if split is None:
        split = tp.split

    sample = tp.problem.sample(K=3, reparam=True, sampling_type=PermutationSampler)

    for (varnames, moment) in tp.moments:
        base_elbo = sample.elbo_vi(split=no_checkpoint)
        test_elbo = sample.elbo_vi(split=split)

        assert t.isclose(base_elbo, test_elbo)

@pytest.mark.parametrize("tp,split", tp_splits)
def test_split_elbo_rws(tp, split):
    """
    tests `sample.elbo_rws` against each other for different splits
    """
    if split is None:
        split = tp.split

    sample = tp.problem.sample(K=3, reparam=False, sampling_type=PermutationSampler)

    for (varnames, moment) in tp.moments:
        base_elbo = sample.elbo_rws(split=no_checkpoint)
        test_elbo = sample.elbo_rws(split=split)

        assert t.isclose(base_elbo, test_elbo)

@pytest.mark.parametrize("tp,split", tp_splits)
def test_split_moments(tp, split):
    """
    tests `marginals.moments` against each other for different splits
    """
    if split is None:
        split = tp.split

    sample = tp.problem.sample(K=3, reparam=False, sampling_type=PermutationSampler)
    base_marginals = sample.marginals(split=no_checkpoint)
    test_marginals = sample.marginals(split=split)

    for (varnames, moment) in tp.moments:
        base_moments = base_marginals._moments(varnames, moment)
        test_moments = test_marginals._moments(varnames, moment)

        assert generic_allclose(base_moments, test_moments)
