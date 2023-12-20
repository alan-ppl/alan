# alan_simplified

To install, navigate to usual directory, and use,
```
pip install -e .
```

### Long-run TODOs:
  * Device (set on Problem).
    - Problem is an nn.Module.
    - Problem has a "dummy" tensor (when you call problem.to('cuda') this dummy tensor gets the same device).
  * Friendly error messages:
    - For mismatching dimension names / plate names for data / inputs / params.
    - Make sure inputs_params have separate names for P and Q.
  * Split for efficient computation.
  * Natural RWS: Bound plate has two extra arguments:
    - a dict of moments + scalar moment initializers {"moment_name": (lambda a, b: a*b, 0.)}
    - a dict telling us how param init + how to convert moments to param {"param_name": lambda mom: mom-3}
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * Timeseries.
  * A better name for BoundPlate.

### Overall workflow design, in terms of user-accessible classes:
  * `Plate` 
    - contains just the definition of P or Q.
    - doesn't contain any inputs or parameters.
  * `BoundPlate`
    - created using `BoundPlate(plate)`
    - binds `Plate`, defining P or Q, to parameters or inputs.
  * `Problem`
    - created using `Problem(P, Q, data, all_platedims)`
    - `P` and `Q` are `Plate`/`BoundPlate`s (any `Plate`s are converted to `BoundPlate` inside Problem).
    - `data: dict[str, torch named Tensor]` (any platedims are named).
    - `all_platedims: dict[str, int]` (size of all platedims).
  * `Sample`
    - created using `problem.sample(K=10)`
    - contains a sample from the approximate posterior.
    - user-facing methods include:
      - `sample.moments`
      - `sample.elbo`
      - `sample.importance_sample` (creates a `ImportanceSample` class)
      - `sample.marginals` (creates a `Marginals` class)
      - no `dump` method, at this stage, as samples here stage aren't user-interpretable.
    - non-user-facing methods:
      - `sample._importance_sample_idxs(N:int, repeats=1)` (Gets the importance-sampled indices).
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
  - `sample.moments("b", Var)`
  - `sample.moments(("a", "b"), Cov)`

For multiple moments, we provide a dict, mapping variable name(s) to moment(s).
```
sample.moments({
    "a": Mean,
    ("a", "b"): Cov
})
```
