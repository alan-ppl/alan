# alan_simplified

To install, navigate to usual directory, and use,
```
pip install -e .
```

TODO:
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * Timeseries.
  * Device (set on Problem).
    - Problem is an nn.Module.
    - Problem has a "dummy" tensor (when you call problem.to('cuda') this dummy tensor gets the same device).
  * Natural RWS: Bound plate has two extra arguments:
    - a dict of moments + scalar moment initializers {"moment_name": (lambda a, b: a*b, 0.)}
    - a dict telling us how param init + how to convert moments to param {"param_name": lambda mom: mom-3}
  * Friendly error messages:
    - For mismatching dimension names / plate names for data / inputs / params.
    - Make sure inputs_params have separate names for P and Q.
  * Split for efficient computation.
  * Make scope simpler once we have sample.
    - Check that the dependency order for P makes sense beforehand.
    - Then just dump the whole of the plate scope directly.
    - As long as inputs_params don't clash for P and Q, then there's only one scope.
