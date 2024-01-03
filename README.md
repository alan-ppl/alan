# alan

Software in development!! Not yet for external use!!

### Installation

Start by removing previous alan or alan_simplified
```
python -m pip uninstall alan
python -m pip uninstall alan_simplified
```
And then remove the corresponding repo folders before cloning the new repo.

To install, clone repo, navigate to repo root directory, and use,
```
pip install -e .
```

### Tests

To run tests, navigate to `tests/` and use `pytest`.

### Docs

[Read the Docs](https://alan-ppl.readthedocs.io/en/latest/)

### Overall example:

See `examples/example.py`



### Meeting TODOs:
  * Docs:
     Friendly overview material (Pyro is a good example).
    - clarify `ExtendedImportanceSample.predictive_ll`.
      - How does it deal with the N samples?
      - How does it deal with the LL for the training data?
  * Think carefully about extended Plate errors.
    - e.g. if you try to extend a prior with plated parameters.
  * Timeseries:
    - tests (Kalman filter)
    - importance_sample
    - extend
  * MovieLens/bus experiments with new code, with VI / RWS / QEM==Natural RWS (see `example/example.py`):
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
   

### Ideas:
  * Can have multiple Timeseries within a group.
```
ab = Group(
    a = Timeseries('a_init', Normal(lambda a: 0.9* a, 0.1))
    b = Timeseries('a_init', Normal(lambda b: 0.9* b, 0.1))
)
```
  * But that means lots of duplicated code e.g. for filtering/resampling the scope in Dist.sample, Timeseries.sample and Group.sample.
  * Solution: function for sample/log_prob that works for Group. Dist / Timeseries call this code, by looking like a one-element Group.  This code handles filtering / resampling the scope, fiddling with the Q log-probs.
    - When combining Group and timeseries, we sample all timesteps of each variable before moviong on to the next variable.
  * `importance_sample.dump` should output tensors with the `N` dimension first?
  * latent moments for `linear_gaussian_latents`
  * tests for mixture distributions.
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
