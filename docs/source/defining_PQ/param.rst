OptParam and QEMParam
====

.. class:: OptParam or QEMParam(init, trans=None, ignore_platenames=(), name=None)

   ``OptParam`` and ``QEMParam`` specify a parameter to be learned (e.g. using VI or QEM).
   
   They don't know the eventual size of the parameter, as they don't know the platesizes.  The actual parameter gets initialized when the Plate is wrapped in a BoundPlate, as that's when the platesizes become known.  Instead, ``OptParam`` or ``QEMParam`` just knows the initial value.


   
   Arguments:
       init (float or Tensor): 
           Initial value of the parameter.  Usually, this would be a float.  If you wanted to specify e.g. the mean of a MultivariateNormal, you'd use a torch.Tensor.  But this would always be a plain tensor, never named.
   
   Keyword Arguments:
       trans (function) (only ``OptParam``): 
           Transformation to be applied to the parameter before use.  This is most useful for e.g. the scale of a Gaussian, which must be postive.  But you can't guarantee that the optimizer will keep the parameter positive.  So instead, you apply a transformation, such as exponentiation, which maps any number on the real line to something positive.  Note that the ``init`` is the initial value _before_ the transformation.  
       ignore_platenames (iterable): 
           By default, we create a parameter with all appropriate platenames.  Such parameters could be quite large, and this could be inappropriate if we want to set up parameter on the prior, P.  So if you want to skip any parameters, you'd include them in here.
       name (str):
           By default, the parameter is named <variable name>_<distribution argument name>.  So in the example above, we'd end up with two parameters, named ``a_loc`` and ``a_scale``.

   Note: No ``transformation`` kwarg for ``QEMParam``
      Parameters learned using QEM don't need transformations to keep them within required limits, so there is no ``transformation`` kwarg.

   Note: QEMParam is all-or-non.
      When using QEM parameters, you must use QEM parameters for all parameter to a distribution, so the following code won't work:

      .. code-block:: python

         P = Plate(
             a = Normal(QEMParam(0.), 1.)
             ...
         )

      You must instead use:

      .. code-block:: python

         P = Plate(
             a = Normal(QEMParam(0.), 1.)
             ...
         )
   
   Note: Names when optimizing parameters in the prior
      When doing Bayesian inference, it is generally preferable to avoid learned parameters in you prior, and instead learn everything.  However, sometimes we do want learned parameters in the prior, and Alan allows this.  However, there is one gotcha.  Namely, that you might want to learn the prior and approximate posterior mean of the random variable a:

      .. code-block:: python

         P = Plate(
             a = Normal(OptParam(0.), 1.)
             ...
         )
         Q = Plate(
             a = Normal(OptParam(0.), 1.)
             ...
         )

      However, this parameter will have the same name (``a_loc``) in P and Q, which isn't allowed.  So we will need to explicitly specify the name on one of these:

      .. code-block:: python

         P = Plate(
             a = Normal(OptParam(0., name="a_loc_P"), 1.)
             ...
         )
         Q = Plate(
             a = Normal(OptParam(0.), 1.)
             ...
         )
