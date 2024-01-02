# alan

[Read the Docs](https://alan-ppl.readthedocs.io/en/latest/)

Software in development!! Not yet for external use!!

### Logistics:

**Cleanup** It is probably best to start by removing previous alan or alan_simplified
```
python -m pip uninstall alan
python -m pip uninstall alan_simplified
```
And then remove the corresponding repo folders before cloning the new repo.

**Installation** To install, clone repo, navigate to repo root directory, and use,
```
pip install -e .
```

**Tests** To run tests, navigate to `tests/` and use `pytest`.

### Recent updates
* For an example, see `examples/example.py`
* Devices should now work.  Just do `problem.to(device='cuda')`, and everything should work without modification.  (Though I have only extensively test sampling).
* Split (an argument to e.g. `sample.elbo`)
* Most of "Overall workflow design" should now be functional
* Three methods for the ELBO/marginal likelihood:
  - `sample.elbo_vi` (reparam=True) 
  - `sample.elbo_rws` (reparam=False, but allows gradients for log-probs)
  - `sample.elbo_nograd` (no gradients at all; useful for memory efficient estimation of marginal likelihood)
* Extensive tests of sample, elbo for splits, devices, plates, unnamed batches and multivariate distributions (e.g. multivariate Normal).  But not extended sampling.
* QEM == Natural RWS.

### Minor TODOs:
  * `importance_sample.dump` should output tensors with the `N` dimension first.
  * `repeats` kwarg for `sample.importance_sample`.
  * check elbo_rws
  * Do we remember what the issue with trailing Ellipsis in e.g. generic_getitem was?
  * latent moments for `linear_gaussian_latents`
  * tests for mixture distributions.
  * tests for histograms from `_marginal_idxs` and `_importance_sample_idxs`
  * remove the random init from most of the tests.
  * tests for extended_sample
    - TestProblem takes extended_platesizes and predicted_extended_moments as arguments.
    - some extended moments are exactly equal to importance sampled moments (i.e. unplated moments + first part of plated moments).
    - other extended moments aren't exactly equal.
    - instead, the extended moments are a function of `importance_sample` moments.
    - TestProblem takes 
    - predicted_extended_moments is a function that takes an importance sample, and returns mean + variance of moment.
    - should really compare moments to 
  * QEM distributions:
    - Categorical
    - Testing

### Meeting TODOs:
  * Docs:
    - clarify `ExtendedImportanceSample.predictive_ll`.
      - How does it deal with the N samples?
      - How does it deal with the LL for the training data?
  * Think carefully about extended Plate errors.
    - e.g. if you try to extend a prior with plated parameters.
  * Friendly docs (Pyro is a good example).
  * Timeseries:
    - tests (Kalman filter)
    - sampling
    - extend
  * MovieLens/bus experiments with new code, with VI, RWS/ Natural RWS (see `example/example.py`):
    - small scale (usual subsampling)
    - large scale using `computation_strategy=Split(...)`
  * User-facing Marginals.ess

### Long-run TODOs:
  * Friendly error messages:
    - Marginals/moments make sense for variables on different plates if they're in the same heirarchy.
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * A `Samples` class that aggregates over multiple `Sample` in a memory efficient way.
    - Acts like it contains a list of e.g. 10 `Sample`s, but doesn't actually.
    - Instead, it generates the `Sample`s as necessary by using frozen random seed.
