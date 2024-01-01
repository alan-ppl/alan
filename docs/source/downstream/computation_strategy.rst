Computation Strategy
====

Many different methods take a ``computation_strategy`` kwarg:

* :func:`Sample.elbo_vi <alan.Sample.elbo_vi>`
* :func:`Sample.elbo_rws <alan.Sample.elbo_rws>`
* :func:`Sample.elbo_no_grad <alan.Sample.elbo_nograd>`
* :func:`Sample.marginals <alan.Sample.marginals>`
* :func:`Sample.importance_sample <alan.Sample.importance_sample>`

This describes the checkpointing/splitting strategy.  

Checkpointing is a technique in neural networks to reduce the memory consumption during backprop.  Usually backprop requires you to save all the intermediate tensors created during the computation.  That can be very memory intensive.  Checkpointing reduces this memory consumption by dropping some of these intermediate tensors, and recreating them during the backward pass.  Thus, while checkpointing can save considerable amounts of memory, it does so by incurring an additional computation cost.

However, for larger models, checkpointing on its own is not sufficient.  In these cases, it may be useful to split along a plate, so that we only compute part of the plate (see :class:`.Split`) below.

The ``computation_strategy`` kwarg can take three values:

* ``alan.no_checkpoint``: all intermediate tensors are saved, potentially requiring lots of memory.
* ``alan.checkpoint``: uses checkpointing to reduce memory consumption.
* ``alan.Split``: uses checkpointing, and further reduces memory consumption by splitting the computation along a plate.

.. autoclass:: alan.Split

