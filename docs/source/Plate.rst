Plate
=====

Defines the structure of the probabilistic program.

.. code-block:: python
    
    model = alan.Model(P, Q())

This handles things such as:
 - Returning samples from the prior and posterior
 - Predictive samples and predictive log likelihood
 - updating parameters for exponential family approximate posteriors (SVI is handled using PyTorch optimizers)
 
.. automodule:: alan.Param
   :members:
   :undoc-members:
   :show-inheritance:
