# alan_simplified

To install, navigate to usual directory, and use,
```
pip install -e .
```

Laurence recent changes:
  * Big changes to ensure we're correctly checking validity of P and Q in Problem, given addition of Data class.
  * I forced `extra_log_factors` to be empty in the sampling stuff, as it doesn't make sense in that context.
  * I stopped the overlap between BoundPlate and Plate.  Now Problem _must_ take a BoundPlate, not a Plate, even if there's no bound inputs/params.  This has likely broken stuff.  But it simplifies things considerably in the long-run.
  * I added a check for the dependency structure in P+Q in Problem (basically, when you set up Problem, you sample from P+Q).

Immanent TODOs:
  * Edits to posterior sampling:
    - I got rid of the mutable `N_dim` on `Sample`, which is really bad; instead that function (now `importance_sampled_idxs`) returns `N_dim`.
    - `sample.index_in` should take `post_idxs, N_dim` as inputs (it knows the sample as it lives inside the object).  You can define this method in terms of a separate function if you like.
    - logPQ_sample should return a dict[str, Tensor], with str representing variable name (mainly because we can always map from str to Dim, but we can't reliably go back).

TODO:
  * Unify scope and input_params when sampling/computing logpq.
  * Device (set on Problem).
    - Problem is an nn.Module.
    - Problem has a "dummy" tensor (when you call problem.to('cuda') this dummy tensor gets the same device).
  * Friendly error messages:
    - For mismatching dimension names / plate names for data / inputs / params.
    - Make sure inputs_params have separate names for P and Q.
  * Make scope simpler once we have sample.
    - Check that the dependency order for P makes sense beforehand.
    - Then just dump the whole of the plate scope directly.
    - As long as inputs_params don't clash for P and Q, then there's only one scope.
  * Split for efficient computation.
  * Natural RWS: Bound plate has two extra arguments:
    - a dict of moments + scalar moment initializers {"moment_name": (lambda a, b: a*b, 0.)}
    - a dict telling us how param init + how to convert moments to param {"param_name": lambda mom: mom-3}
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * Timeseries.
