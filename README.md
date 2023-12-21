# alan_simplified
To install, clone repo, navigate to repo root directory, and use,
```
pip install -e .
```

### Recent updates

* Devices should now work.  Just do `problem.to(device='cuda')`, and everything should work without modification.  (Though I have only extensively test sampling).
* Split (an argument to e.g. `sample.elbo`)
* Most of "Overall workflow design" should now be functional

### Minor TODOs:
  * Marginals make sense for variables on different plates if they're in the same heirarchy.
  * `importance_sample.dump` should output tensors with the `N` dimension first.
  * Implement `sample.moments` for `CompoundMoment`.
  * `repeats` kwarg for `sample.importance_sample`.
  * use `PermutationMixtureSample` as the default `SamplingType`.

### Long-run TODOs:
  * Friendly error messages:
    - For mismatching dimension names / plate names for data / inputs / params.
    - Make sure inputs_params have separate names for P and Q.
  * Natural RWS: Bound plate has two extra arguments:
    - should be able to implement it in terms of `sample.moments`
    - provide a dict to `BoundPlate`, mapping variable name {'a': NaturalRWS(init_mean = 0., init_scale=1.)}
    - assume (and check) that variables for which we're doing natural RWS are written as `a = Normal('a_mean', 'a_scale')` (i.e. the parameters are specified as strings).
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * Timeseries.
  * A better name for BoundPlate.
  * A `Samples` class that aggregates over multiple `Sample` in a memory efficient way.
    - Acts like it contains a list of e.g. 10 `Sample`s, but doesn't actually.
    - Instead, it generates the `Sample`s as necessary by using frozen random seed.

### Overall workflow design, in terms of user-accessible classes:
  * `Plate` 
    - contains just the definition of P or Q.
    - doesn't contain any inputs or parameters.
  * `BoundPlate`
    - created using `BoundPlate(plate)`
    - binds `Plate`, defining P or Q, to parameters or inputs.
    - user-facing methods include:
      - `bound_plate.sample`
  * `Problem`
    - created using `Problem(P, Q, data, all_platesizes)`
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
  - `sample.moments("b", (Mean, Var))`
  - `sample.moments(("a", "b"), Cov)`

For multiple moments, we provide a dict, mapping variable name(s) to moment(s).
```
sample.moments({
    "a": Mean,
    "b": (Mean, Var),
    ("a", "b"): Cov
})
```


