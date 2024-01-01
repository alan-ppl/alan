Distributions
=====

Alan wraps all the distributions in ``torch.distributions``, so for all the distributions in ``torch.distributions``, we have a corresponding distribution with exactly the same signature.


The only different thing in Alan is that distributions can take a number of things as arguments:

* A number / tensor (representing a fixed parameter)
* A string (usually representing another random variable)
* A function (usually representing a transformation of another random variable)
* OptParam or QEMParam (representing a learned parameter)

Critically, the string, or __arguments__ to the function must be:

* A previously sampled random variable.
* An input provided to :class:`.BoundPlate`.
* An extra_opt_param provided to :class:`.BoundPlate`.


