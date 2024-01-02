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
  * User-facing ESS
  * Errors:
    - No Data / Enum in P.

### Meeting TODOs:
  * Error if you try to extend a prior with plated parameters.
  * Clarify `ExtendedImportanceSample.predictive_ll`.
    - How does it deal with the N samples?
    - How does it deal with the LL for the training data?
  * Friendly docs (Pyro is a good example).
  * Timeseries:
    - tests
    - sampling
    - extend

### Long-run TODOs:
  * Friendly error messages:
    - Marginals/moments make sense for variables on different plates if they're in the same heirarchy.
  * Enumeration:
    - Enumeration is a class in Q (like Data), not P.
  * A `Samples` class that aggregates over multiple `Sample` in a memory efficient way.
    - Acts like it contains a list of e.g. 10 `Sample`s, but doesn't actually.
    - Instead, it generates the `Sample`s as necessary by using frozen random seed.
  * Timeseries (see below)

### Timeseries plan:
* Timeseries acts as a replacement for a Dist.
```
plate = Plate(
    T = Plate(
        a = Timeseries(...)
        bc = Group(
            b = Timeseries(...),
            c = Timeseries(...),
        )
    )
)
```
* Note that if there's dependencies across timeseries (say, timeseries named a and b, where b depends on a), then b_t depends on a_t, not a_{t-1}. (That's consistent with what has to happen if e.g. you have a timeseries depending on a variable that's independent across the plate, or vice-versa).
* Two timeseries `log_pq` algorithms:
  - initial state 
  - strategy 1: 
    - do reduce_Ks on everything to get a single enormous T x (K_timeseries...) x (K_timeseries...) tensor.
    - do a chain matmul
    - few tensor operations (because of chain matmul), but lots of memory (K^{2N}), where N is the number of timeseries variables.
    - makes sense with split, because we can easily return and combine T x (K_timeseries...) x (K_timeseries...) tensors.
    - initial log_prob is annoying, but can be dealt with.
    - Need to introduce extra K_timeseries dimensions for log_prob.  As these are immediately summed out, I think they can be created + destroyed locally, and don't need to be managed globally.
    - At the final step, log_PQ returns a tensor with just a single K_dimension (as we've summed out the initial state).
    - Detail alg:
      - All log_PQ methods return tensor, prev_Ks, curr_Ks.
        - Mostly, prev_Ks, curr_Ks are empty tuples.
      - Go though all variables in a plate, collecting log_PQ, prev_Ks, curr_Ks.
      - Sum out Ks that aren't in the previous plate, prev_Ks or curr_Ks.
      - Gives one big tensor you can chain_matmul.
      - Split strategy:
        - Split the timeseries samples along T.
        - But then you want the initial sample to be the same K-dimension as the rest of the timeseries.
        - Give plate_log_PQ samples of length split_T, get back K_start x K_end.
  - strategy 2:
    - But you can actually do quite a bit better than strategy 1 in terms of asymptotic complexity, though at the cost of far, far more tensor ops.
    - This is evident if you consider using the usual within-plate reduction strategy. 
    - In particular, the most efficient approach is to sum out variables, starting at the end and working backwards.
    - Requries more complex implementation + far more tensor ops, but gives asymptotic reductions in complexity.
