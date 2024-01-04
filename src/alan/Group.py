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

        #Finalize the distributions by passing in the varname, and check types
        self.prog = {varname: dist.finalize(varname) for (varname, dist) in kwargs.items()}

        set_all_arg_list = set([arg for dist in self.prog.values() for arg in dist.all_args])
        self.all_args = set_all_arg_list.difference(self.prog.keys()) #remove dependencies on other variables in the group.

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            inputs_params: dict,
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampler:Sampler,
            reparam:bool,
            ):

        return sample_gdt(
            prog=self.prog,
            scope=scope,
            K_dim=groupvarname2Kdim[name],
            groupvarname2Kdim=groupvarname2Kdim,
            active_platedims=active_platedims,
            sampler=sampler,
            reparam=reparam,
        )
    
    def sample_extended(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:dict):
        
        result = {}       #This is the sample returned.

        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)

        # Loop through all dists in the group and sample from them (plus potentially get 
        # logprobs of original and extended data IF extended_data is provided, i.e. not None)
        for name, dist in self.prog.items():

            childsample = dist.sample_extended(
                sample=sample.get(name),
                name=name,
                scope=scope,
                inputs_params=inputs_params,
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                active_extended_platedims=active_extended_platedims,
                Ndim=Ndim,
                reparam=reparam,
                original_data=original_data,
            )

            scope[name]  = childsample
            result[name] = childsample

        return result
    
    def predictive_ll(
        self,
        sample:dict,
        name:str,
        scope:dict[str, Tensor],
        inputs_params:dict,
        original_platedims:dict[str, Dim],
        extended_platedims:dict[str, Dim],
        original_data: dict[str, Tensor],
        extended_data: dict[str, Tensor]):
        
        #resampled scope is the scope used in here when sampling from the Group
        scope = self.filter_scope(scope)

        original_lls, extended_lls = {}, {}

        # Loop through all dists in the group and sample from them (plus potentially get 
        # logprobs of original and extended data IF extended_data is provided, i.e. not None)
        for name, dist in self.prog.items():

            child_original_lls, child_extended_lls = dist.predictive_ll(
                sample=sample.get(name),
                name=name,
                scope=scope,
                inputs_params=inputs_params,
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                original_data=original_data,
                extended_data=extended_data
            )

            scope[name]  = sample.get(name)

            original_lls = {**original_lls, **child_original_lls}
            extended_lls = {**extended_lls, **child_extended_lls}

        return original_lls, extended_lls


    def all_prog_names(self):
        return self.prog.keys()
