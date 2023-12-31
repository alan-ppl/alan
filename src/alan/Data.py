class Data():
    """
    alan.Data()
    Used to specify a variable names that will be associated with data in the approximate posterior, Q.

    Usually, the prior, P, would specify the distribution over data, while 
    the approximate posterior, Q, wouldn't.  However, that causes problems,
    because it means that P and Q have different variables, and potentially
    even different plates (e.g. if there is a plate just for data).  That
    makes it difficult and error prone e.g. to check that P and Q are
    compatible with each other.

    To fix that problem, when there is data in Q that you would usually omit.
    For instance, in the following code, ``b`` is given as data, so we use
    ``b = Data()`` in ``Q``:

    .. code-block:: python

       P = Plate(
          a = Normal(0., 1.),
          b = Normal('a', 1.),
       )
       Q = Plate(
          a = Normal(0., 1.),
          b = Data(),
       )

    Note that ``Data()`` never takes any arguments.
    """
