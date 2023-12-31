# alan

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
  * consider adding .sample_reparam and .sample_non_reparam to Sample (Sample with reparam=True has both).
  * Do we remember what the issue with trailing Ellipsis in e.g. generic_getitem was?
  * latent moments for `linear_gaussian_latents`
  * tests for mixture distributions.
  * tests for histograms from `_marginal_idxs` and `_importance_sample_idxs`
  * make sure most of the tests have a split specified.
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


### Long-run TODOs:
  * Friendly error messages:
    - Marginals/moments make sense for variables on different plates if they're in the same heirarchy.
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * Timeseries:
    - Lives within a plate; can only be the first thing in a plate.
    - log prob algorithm:
      - sum out all the other K's on the plate, but keep `K_timeseries`.
      - sum out the plate by handing sample of timeseries + log prob with everything else summed out to a method on Timeseries.
    - Syntax:
```
plate = Plate(
    T = Plate(
        timeseries = Timeseries(
            initial_dist = Normal(0, 1),
            transition = lambda x: Normal(0.9*x, 0.1),
        )
        noisy_timeseries = Normal('timeseries', 0.3),
    )
)
```
  * A `Samples` class that aggregates over multiple `Sample` in a memory efficient way.
    - Acts like it contains a list of e.g. 10 `Sample`s, but doesn't actually.
    - Instead, it generates the `Sample`s as necessary by using frozen random seed.

### Overall workflow design, in terms of user-accessible classes:
For an example, see `examples/example.py`

  * `OptParam` / `QEMParam`
    - Used as a direct argument to a distribution (e.g. Normal(OptParam(1.), OptParam(1.))) inside the `Plate`.
    - The first argument (1. above) is the initial value.
    - Specifies that a parameter should be created.
  * `Plate` 
    - contains just the definition of P or Q.
    - doesn't know the platesizes, so cannot e.g. sample or initialize parameters.
  * `BoundPlate`
    - created using `BoundPlate(plate, all_platesizes)`
    - all_platesizes is a dict mapping platename -> int.
    - initializes and stores all the parameters.
    - does know the platesizes, so can be sampled.
    - Has optional arguments for `inputs` + `extra_opt_params`.  These are provided as a dict, mapping inputname/paramname -> named Tensor.
    - user-facing methods include:
      - `bound_plate.sample()` Draws a single sample from the distribution (e.g. to sample data from a generative model).
      - `bound_plate.opt_params()` Returns a dict of all the optimized parameters.
      - `bound_plate.inputs()` Returns a dict of all the inputs.
      - `bound_plate.qem_params()` Returns a dict of all the QEM-learned parameters.
      - `bound_plate.qem_means()` Returns a dict of all the means used in QEM.
  * `Problem`
    - created using `Problem(P, Q, data)`
    - `P` and `Q` are `BoundPlate`s.
    - `data` is a dict mapping the dataname to a named Tensor.
    - user-facing methods include:
      - `problem.sample(K=10)`: produces a `Sample`.
  * `Sample`
    - created using `problem.sample(K=10)`
    - contains a sample from the approximate posterior.
    - user-facing methods include:
      - `sample.moments`
      - `sample.importance_sample` (creates a `ImportanceSample` class)
      - `sample.marginals` (creates a `Marginals` class)
      - `sample.qem_update_params(lr)` (updates QEM params)
      - Three methods for the ELBO/marginal likelihood:
        - `sample.elbo_vi` (reparam=True) 
        - `sample.elbo_rws` (reparam=False, but allows gradients for log-probs)
        - `sample.elbo_nograd` (no gradients at all; useful for memory efficient estimation of marginal likelihood)
      - no `dump` method, at this stage, as samples here stage aren't user-interpretable.
  * `ImportanceSample`
    - created using `sample.importance_resample(N, repeats=1)`.
    - `repeats` runs a for loop to collect more samples.
    - contains a reference to `Problem`, _not_ `Sample`.
    - has already done `index_in`: just has samples, not indicies to samples.  This is nice because it means that `ImportanceSample`s arising from different `Sample`s can easily be combined, and its what we need for `ExtendedImportanceSample`.
    - methods including:
      - `importance_sample.extend`
      - `importance_sample.dump` (gives a user-readable flat dict of samples)
      - `importance_sample.moments`
  * `ExtendedImportanceSample`
    - created using `importance_sample.extend(all_extended_platedims)`
    - samples the extra stuff from the prior, but doesn't (yet) compute e.g. predicted LL.
    - extends everything using `P`, including the data (predictions for data in the future are likely useful in and of themselves).
    - methods including:
      - `extended_importance_sample.predictive_ll`
      - `extended_importance_sample.dump` (gives a user-readable flat dict of samples).
      - `extended_importance_sample.moments`
  * `Marginals`
     - created using `sample.marginals`.
     - By default contains all univariate marginals.  But may also contain multivariate marginals.
     - Marginals stored as joint distributions over `K` computed using the source-term trick.
     - Makes it efficient to compute one moment at a time (whereas e.g. for `Sample`, we need to compute all the log-probs and backprop to compute a moment, so it is better to compute lots of moments together).
     - methods including:
       - `marginals.moments`

### Interface for moments:
See `moments.py`.  The basic idea is that we should have a uniform way of calling:
 - `sample.moments`
 - `marginals.moments`
 - `importance_sample.moments`

Specifically, each of these methods can be called, with variable name(s) as a string/tuple of strings, and moments as a class:
  - `sample.moments("a", Mean)`
  - `sample.moments(("a", "b"), Cov)`

For multiple moments, we provide a dict, mapping variable name(s) to moment(s).
```
sample.moments([
    "a": Mean,
    ("a", "b"): Cov
])
```


