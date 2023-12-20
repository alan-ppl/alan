# alan_simplified

To install, navigate to usual directory, and use,
```
pip install -e .
```

TODO (long run):
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

Interface design:
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
    - methods include:
      - `sample.marginals`
      - `sample.moments`
      - `sample.importance_resample`
      - `sample.elbo`
      - `sample.dump` (gives a user-readable flat dict of samples).
  * `ImportanceSample`
    - created using `sample.importance_resample(N)`.
    - contains a reference to `sample`.
    - methods including:
      - `importance_sample.extend`
      - `importance_sample.dump` (gives a user-readable flat dict of samples)
  * `ExtendedImportanceSample`
    - created using `importance_sample.extend(all_extended_platedims)`
    - samples the extra stuff from the prior, but doesn't (yet) compute e.g. predicted LL.
    - extends everything using `P`, including the data (predictions for data in the future are likely useful in and of themselves).
    - methods including:
      - `extended_importance_sample.predictive_ll`
      - `extended_importance_sample.dump` (gives a user-readable flat dict of samples).


