Moments
====

Four classes have a ``moments`` method: 

* :class:`.Sample`
* :class:`.Marginals`
* :class:`.ImportanceSample`
* :class:`.ExtendedImportanceSample`

They all work the same way, computing posterior moments using some reweighting strategy.
Specifically, 
* importance weighting for :class:`.Sample` and :class:`.Marginals`
* importance sampling for :class:`.ImportanceSample` and :class:`.ExtendedImportanceSample`)

Note that computing moments using importance weighting (i.e. from :class:`.Sample` and :class:`.Marginals`) is always going to be more accurate than computing moments from importance samples (i.e. from :class:`.ImportanceSample`).

There are several different ways to call obj.moments:

.. function:: obj.moments(varname: str, moment: alan.Moment)

   The most basic form of the moments function, which computes a single, univariate moment.

   Example:
      .. code-block:: python

         sample.moments('a', alan.mean)

   Arguments:
      varname (str): 
         the name of the variable to compute the moment.
      moment (alan.Moment):
         the moment to compute.

   Returns:
      the moment represented an named ``torch.Tensor``, where the names correspond to the plates.


.. function:: obj.moments(varnames: tuple[str], moment: alan.Moment)

   Some moments (such as a covariance) depend on two random variables.  This form admits tuples of random variable names, allowing you to compute a single, multivariate moment.

   Example:
      .. code-block:: python

         sample.moments(('a', 'b'), alan.cov)

   Arguments:
      varname (tuple[str]): 
         a tuple of names of the variable.
      moment (alan.Moment):
         the moment to compute.

   Returns:
      the moment represented an named ``torch.Tensor``, where the names correspond to the plates.


.. function:: obj.moments(moment_list: list[tuple[varname(s), alan.Moment]])

   If you're computing moments from :class:`.Sample`, then it is far more efficient to pass in a whole bunch of moments at once.  This form of the moments function allows you to do that.

   Example:
      .. code-block:: python

         sample.moments([
             ('a', mean),
             ('b', mean),
         ])


   Arguments:
      moment_list (list[tuple[varname(s), alan.Moment]]):
         A list of tuples with two elements.  The first element is the variable name (str) or the variable names (tuple[str]), and the second element is an alan.Moment.

   Returns:
      A list of moments represented an named ``torch.Tensor``, where the names correspond to the plates.

When we're specifying the moment using an ``alan.Moment`` class, we have a few options:
* ``mean``: computes the mean
* ``mean2``: computes the raw second moment
* ``var``: computes the variance
