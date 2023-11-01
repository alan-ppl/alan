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
        a = alan.Normal(0,   1),
        b = alan.Normal('a', 1),
    )
    plate1 = Plate(
        c = alan.Normal('b', 1),
        d = alan.Normal('c', lambda c: c.exp())
        plate2 = Plate(
            e = alan.Normal('c', scale=1),
            f = alan.Normal('e', scale=1),
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
  - a lambda `lambda a: a.exp()`.  Note that the identity of the input variable is determined by the argument name.
* Variables that are in scope are: 
  - those previously generated in the current plate
  - those previously generated in higher-level plates
  - fixed inputs (e.g. film features in MovieLens, or approximate posterior parameters)
* When we run e.g. `prog.sample`, we provide:
  - plate sizes.
  - inputs.
* samples are represented as dictionaries (we will allow for e.g. timeseries or GPs, but they should be interchangeable: i.e. you can have a timeseries in the generative model, but a plate in the approximate posterior).
* backend uses named tensors (so we don't have to worry about named dimensions).

TODOs (B/T):
* Implement and test SamplingType.
* Tests for TorchDimDist.
  - note that sample_dims is supposed to be all the samples in the output, not just the extra ones.
* Tests for LP_Plate (explicitly construct some LP_Plates, and sum out.  Do we get what we expect?)

TODOs (L):
* Check that all names are unique as you construct plate.
* Source term tricks
* Groups
* Enumerating discrete variables
* Syntax for Timeseries


Groups:
  * groups are denoted in programs using ab = Group(a=alan.Normal(0,1), b=alan.Normal('a', 2))
  * groups only appear in Q (as they're all about sampling the approximate posterior, they don't make sense in P).
  * groups can't have inner plates/timeseries.
  * the sample from the group are GroupSample (which are just a thin wrapper over a dict).
  * but log_prob from groups are aggregated across all variables within the group (log_prob for P knows it needs to aggregate because it sees the GroupSample).

Efficiency:
  * except for very, very large models, it should always be possible to store the samples in (video) RAM.
  * what is hard is to represent the log-prob tensors that arise as we're summing out K's.
  * the solution is to split this sum over a plate.
  * specifically, write down a function that takes:
    - full Q_sample
    - Q_part, P_part (i.e. part of the Q and P models, perhaps corresponding to the lowest-level plate).
    - now we can split the sum along the plate into parts, and do each part in turn.
  * this function is "checkpointed" for the purposes of the backward pass (but might have to manually checkpoint).

Combining prog with inputs/learned parameters/data.
  * inputs / learned parameters are just dumped in scope.
  * data is added to a sample before we compute log P.
  * Assumptions:
    - Q just has learned parameters (no inputs or data).
    - P just has inputs in scope (no data or learned parameters).
    - data is only added to the sample from Q
  * Solution: a Bind(plate, inputs=None, parameters=None) method.
    - inputs are fixed and not learned (used for fixing inputs for P)
    - parameters are learned (used for fixing inputs for Q)
    - inputs and parameters are named tensors, not torchdim yet (as we haven't associated P with Q, we can't have a unified list of Dims).
  

Datastructures:
  * Program is represented as a nested series of objects (mainly Plate)
  * samples are represented as a nested dict of:
    - torchdim tensors.
    - GroupSample (for groups) which acts as a thin wrapper.
  * log-probs are represented as a nested series of objects (mainly LP_Plate).  This is for a few reasons:
    - we can use methods on these objects to compute the ELBO, without needing the original program.
    - but those methods need to know e.g. Timeseries vs Plate.
  * scope is used as we're going through a program (e.g. sampling or computing log_prob), and is a flat dict representing all the accessible variables.

