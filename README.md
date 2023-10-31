# alan_simplified

To install, navigate to usual directory, and use,
```
pip install -e .
```

After thinking hard about our "overlapping plate" issue, I realise that the class of model we can apply massively parallel methods to is much smaller than we originally thought. Specifically, that plates must be nested, they can't "cross over".

This leads to quite strong restrictions.
While that might sound like a bad thing, in some ways it is nice.
Those restrictions make it much easier to write a PPL that is correct, because far fewer things can happen.

Specifying a program will look something like:
```
prog = Plate(
    ab = Group(
        a = alan.Normal(loc=0, scale=1),
        b = alan.Normal(loc='a', scale=1),
    )
    plate1 = Plate(
        c = alan.Normal(loc='b', scale=1),
        d = alan.Normal(loc='c', scale=lambda tr: tr['c'].exp()),
        plate2 = Plate(
            e = alan.Normal(loc='c', scale=1),
            f = alan.Normal(loc='e', scale=1),
        ),
    ),
)
```
Some observations here:
* This code doesn't actually run the probabilistic program.  Instead, it just defines the program, as a nested series of `Plate` objects.
* At the top layer, there is an unnamed plate.
* Plates contain variables and other plates.
* Distributions have three types of input args:
  - specific values (e.g. scalars).
  - names of previous values.
  - a lambda `lambda tr: tr['a'].exp()`, which takes `tr`, a dict containing all variables that are "in scope".
* Variables that are in scope are: 
  - those previously generated in the current plate
  - those previously generated in higher-level plates
  - fixed inputs (e.g. film features in MovieLens, or approximate posterior parameters)
* When we run e.g. `prog.sample`, we provide:
  - plate sizes.
  - inputs.
* samples are represented as dictionaries (we will allow for e.g. timeseries or GPs, but they should be interchangeable: i.e. you can have a timeseries in the generative model, but a plate in the approximate posterior).
* backend uses named tensors (so we don't have to worry about named dimensions).

TODOs (immanent):
* Implement and test SamplingType.
* Tests for TorchDimDist.
  - note that sample_dims is supposed to be all the samples in the output, not just the extra ones.

TODOs (pending):
* Do subtracting logK in logP.



Components:
*TorchDimDist

Algorithm:
* Sample from Q with a single K-dimension:
  - Needs to know K-dimension.
  - May permute/resample the input variables, depending on the scheme.
  - Don't sample enumerate variables.
* Convert single K-dimension to multiple K-dimensions.
* Compute logQ.
  - Need to know the K-dimension associated with each variable.
  - Needs to know the resampling scheme.
* Compute logP (log_prob doesn't need to know K-dimensions).
* Take difference logP - logQ
* Reduce

Groups:
  - groups are denoted in programs using ab = Group(a=alan.Normal(0,1), b=alan.Normal('a', 2))
  - groups must match across P and Q (otherwise e.g. doing logP-logQ becomes painful).
  - groups can't have inner plates/timeseries.
  - samples from groups are just dicts (like plate samples).
  - but log_prob from groups are aggregated across all variables within the group.

Datastructures:
  * Program is represented as a nested series of objects (mainly Plate)
  * samples are represented as a nested dict of torchdim tensors
    - nesting for groups/plates/timeseries.
    - means that it isn't possible to tell the difference between group/plate/timeseries based only on sample.
  * log-probs are represented as a nested series of objects (mainly LP_Plate).  This is for a few reasons:
    - we can use methods on these objects to compute the ELBO, without needing the original program.
    - but those methods need to know e.g. Timeseries vs Plate.
  * scope is used as we're going through a program (e.g. sampling or computing log_prob), and is a flat dict representing all the accessible variables.

