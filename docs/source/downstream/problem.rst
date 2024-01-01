Problem
==== 

.. autoclass:: alan.Problem

   .. automethod:: sample

   .. method:: to(device='cpu')

   This method is inherited from ``torch.nn.Module``, and allows you to put the problem on a device.
   Once you've done that, all the parameters and subsequent computation should be on the device.

   Example:
      .. code-block:: python
      
         problem = Problem(P, Q, all_platesizes)
         problem.to(device='cuda')
