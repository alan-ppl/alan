import torch as t

from typing import Optional
from .dist import _Dist, sample_gdt
from .utils import *
from .Sampler import Sampler


class Group(): 
    """
    A class used when defining the model in order to speed up inference.

    Alan fundamentally works by drawing K samples of each latent variable, and considering all possible combinations of those variables.  It sounds like this would be impossible, as there is K^n combinations, where n is the number of latent variables.  Alan circumvents these difficulties using message passing-like algorithms to exploit conditional indepdencies, and get computation that is polynomial in K.  However, the complexity can still be K^3 or K^4, and grouping variables helps to reduce that power.

    In particular, consider two models

    Slower model (K^3):

    .. code-block:: python

       Plate(
           loc       = Normal(0., 1.),
           log_scale = Normal(0., 1.),
           d = Normal(loc, lambda log_scale: log_scale.exp())
           ...
       )

    This model has K^3 complexity, as we will need to compute the log-probability of all K samples of ``d`` for all K^2 samples of ``loc`` and ``log_scale``.  (There's K samples of ``loc`` and K samples of ``log_scale``, so K^2 combinations of samples of ``loc`` and ``log_scale``).
    That K^3 complexity is excessive for this simple model.
    One solution would be to not consider all K^2 combinations of ``loc`` and ``log_scale``, but instead consider only the K corresponding samples.  That's precisely what ``Group`` does:

    Faster model (K^2):

    .. code-block:: python

       Plate(
           g = Group(
               loc       = Normal(0., 1.),
               log_scale = Normal(0., 1.),
           ),
           d = Normal(loc, lambda log_scale: log_scale.exp())
           ...
       )

    The arguments to group are very similar to those in :class:`.Plate`, except that you can only have distributions, not sub-plates, sub-groups or :class:`.Data`.
    """
    def __init__(self, **kwargs):
        #Groups can only contain Dist, not Plates/Timeseries/Data/other Groups.
        for varname, dist in kwargs.items():
            if not isinstance(dist, (_Dist)):
                raise Exception("{varname} in a Group should be a Dist or Timeseries, but is actually {type(dist)}")

        if len(kwargs) < 2:
            raise Exception("Groups only make sense if they have two or more random variables, but this group only has {len(kwargs)} random variables")

        self.prog = {varname: dist.finalize(varname) for (varname, dist) in kwargs.items()}
