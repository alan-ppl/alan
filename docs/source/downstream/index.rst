Downstream
=====

Once we have defined P and Q using ..., we need to actually do stuff with these distributions.

At a high level, the flow is:

* Define the prior, ``P_plate``, and approximate posterior, ``Q_plate``, abstractly as ``Plate``'s.  Note that these do not currently have platesizes, so they can't be initialized, and we haven't e.g. concretely initialized any parameters.
* Bind the prior and approximate posterior to platesizes and initialized the parameters, by passing ``P_plate`` and ``Q_plate``, to ``BoundPlate``.
* Construct an inference problem, by passing ``P_bound_plate`` and ``Q_bound_plate`` into ``Problem``, along with a dict of data.
* Call ``problem.sample(K=10)``, which draws 10 samples for each random variable.
* Do something with the sample (there's lots of options, see below).

A code example for this flow is:

.. code-block:: python

   import alan

   P_plate = alan.Plate( 
       a = alan.Normal(0., 1),
       p1 = alan.Plate(
           b = alan.Normal('a', 1),
           p2 = alan.Plate(
               c = alan.Normal("b", 1),
           ),
       ),
   )

   Q_plate = alan.Plate( 
       a = alan.Normal(0., 1),
       p1 = alan.Plate(
           b = alan.Normal('a', 1),
           p2 = alan.Plate(
               c = alan.Data(),
           ),
       ),
   )

   all_platesizes = {'p1': 3, 'p2': 4}
   P_bound_plate = alan.BoundPlate(P_plate, all_platesizes)
   Q_bound_plate = alan.BoundPlate(Q_plate, all_platesizes)

   data = {'c': t.randn((3,4), names=('p1', 'p2')}
   problem = alan.Problem(P_bound_plate, Q_bound_plate, data)

   sample = data.sample(K=10)

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   bound_plate
   problem
   sample
   marginals
   importance_sample
   extended_importance_sample
   moments
   computation_strategy
