Defining P and Q
=====

Here, we document all the alan functionality you need to define the structure of the prior, P, and approximate posterior, Q.

.. code-block:: python

   from alan import Plate, Normal, Group, Data, QEMParam, OptParam
   
   P_plate = Plate( 
       a = Normal(0., 1),
       bc = Group(
           b = Normal('a', 1),
           c = Normal('b', lambda a: a.exp()),
       ),
       p1 = Plate(
           d = Normal("c", 1),
           e = Normal("d", 1.),
       ),
   )

   Q_plate = Plate( 
       a = Normal(QEMParam(0.), QEMParam(1.)),
       bc = Group(
           b = Normal(QEMParam(0.), QEMParam(1.)),
           c = Normal(0., 1.),
       ),
       p1 = Plate(
           d = Normal(OptParam(0.), OptParam(1.)),
           e = Data(),
       ),
   )
    
.. autoclass:: alan.Plate
   :members: sample

.. autoclass:: alan.Group

.. autoclass:: alan.Data

.. autoclass:: alan.OptParam

.. autoclass:: alan.QEMParam

Distributions...
