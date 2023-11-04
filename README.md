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

TODOs (L):
* Checking:
  - check scoping
  - check platedims with inputs, parameter and data.
* Check reduce_logQ for IndependentSample
* Source term tricks
* Enumerating discrete variables
* Syntax for Timeseries

logPQ:
* The "magic" happens in here.
* Note that this is structured the way it is for efficiency:
  - logPQ_plate takes P, Q, samples etc for a sub-plate of the program, and returns a single tensor (summing out K and the platedim).
  - this function can easily be split across the plate and checkpointed.
* P, Q are nested dicts containing the prior and approximate posterior.  Structure must match exactly.
* sample_data has the same structure as P.
* inputs_params is a nested dict, with the same "Plating" structure as P, Q or sample_data.
* extra_log_factors is a nested dict, with the same "Plating" structure as P, Q or sample_data.
* Using the source-term trick to compute moments or importance weights is easy enough.  Computing the conditionals for sampling is much more painful:
  - Use standard parallelisation tricks to sample from the unsplit latents at the top.
  - Then recompute the split log-prob, fixing the split latents at the top.

Sampling:
  * When sampling there's always a K-dimension.
  * The K-dimension could be of size 1 though.

Principles:
  * P has exactly the same structure as Q, except that Q lacks data.
  * so Q.sample + data has the same structure as P.sample.
  * Groups:
    - groups are denoted in programs using ab = Group(a=alan.Normal(0,1), b=alan.Normal('a', 2))
    - matching groups must appear in both P and Q
    - groups can't have inner plates/timeseries
    - groups can't have data.
  * Enumerate:
    - to sum over a discrete latent variable, we include a "fake distribution", alan.Enumerate in Q.
    - the sample from this distribution just enumerates all possible settings.
    - the log-prob is logQ = - log(K) (so that it cancels when we do logP-logQ-log(K))

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

Enumeration:
  * Enumerate type in Q.
  

Datastructures:
  * Program is represented as a nested series of objects (mainly Plate)
  * samples are represented as a nested dict of:
    - torchdim tensors.
    - GroupSample (for groups) which acts as a thin wrapper.
  * log-probs are represented as a nested series of objects (mainly LP_Plate).  This is for a few reasons:
    - we can use methods on these objects to compute the ELBO, without needing the original program.
    - but those methods need to know e.g. Timeseries vs Plate.
  * scope is used as we're going through a program (e.g. sampling or computing log_prob), and is a flat dict representing all the accessible variables.

