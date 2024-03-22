import torch as t
from typing import Optional

from functorch.dim import Dim

from .utils import *
from .Sampler import Sampler
from .dist import Dist
from .Group import Group
from .Data import Data
from .dist import Dist, _Dist, sample_gdt, datagroup
from .Timeseries import Timeseries



class Plate():
    """
    The key class used to define your model: all random variables, are defined within a ``Plate``.

    An example plate definition:

    .. code-block:: python

       Plate(
           a = Normal(0., 1.),
           g = Group(
               b = Normal('a', 1.),
               c = Normal('b', 1.),
           ),
           p = Plate(
               d = Data(),
           ),
       )

    Everything in the plate is specified as a keyword argument (i.e. of the form ``name = thing``), where ``thing`` could be:
    
    * a distribution (see :ref:`Distributions`).
    * a :class:`.Group`.
    * a sub-plate Plate.
    * :class:`Data() <.Data>`.

    Critically, the name becomes the name of that thing.  So in the above example, we have normal random variables named ``a``, ``b``, and ``c``, a group named ``g``, a sub-plate named ``p``, and a random variable that will be associated with data, (see :class:`.Data`).

    Note: 
        In standard Bayesian terminology, including a variable within a plate indicates that there is actually several of these variables.  That's precisely how we're using sub-plates: each subplates (``p`` in the example) with have an assigned platesize, and we'll replicate each variable within the plate that number of times.  

        Note that we also use ``Plate`` at the top-layer, even though we only have one copy of the top-layer variables.

    """
    def __init__(self, **kwargs):

        #Finalise any dists
        kwargs = {k: v.finalize(k) if isinstance(v, _Dist) else v for (k, v) in kwargs.items()}

        self.grouped_prog = {}
        self.flat_prog = {}
        self.groups = {}
        for k, v in kwargs.items():
            if isinstance(v, Plate):
                self.grouped_prog[k] = v
                self.flat_prog[k] = v
            else:
                assert isinstance(v, (Group, Dist, Timeseries, Data))

                if isinstance(v, Group):
                    group = v.prog
                    self.groups[k] = v
                else:
                    group = {k: v}

                self.grouped_prog[k] = {}
                for gk, gv in group.items():
                    self.grouped_prog[k][gk] = gv
                    self.flat_prog[gk] = gv


        #Error checking: plate/variable/group names aren't reserved
        all_prog_names = self.all_prog_names()
        for name in all_prog_names:
            check_name(name)

        #Error checking: no duplicate names.
        dup_names = list_duplicates(all_prog_names)
        if 0 != len(dup_names):
            raise Exception(f"Plate has duplicate names {dup_names}.")

    def grouped_get(self, d, groupname):
        gv = self.grouped_prog[groupname]
        if isinstance(gv, dict):
            return {k: d.get(k) for k in gv}
        else:
            assert isinstance(gv, Plate)
            return d[groupname]

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

        if name is not None:
            active_platedims = [*active_platedims, all_platedims[name]]

        scope = update_scope(scope, inputs_params)
        sample = {}
        
        for childname, prog in self.grouped_prog.items():
            if isinstance(prog, dict):
                if not datagroup(prog):
                    childsample = sample_gdt(
                        prog=prog if isinstance(prog, dict) else {name: prog},
                        scope=scope,
                        active_platedims=active_platedims,
                        K_dim=groupvarname2Kdim[childname],
                        groupvarname2Kdim=groupvarname2Kdim,
                        sampler=sampler,
                        reparam=reparam
                    )

                    for k, v in childsample.items():
                        sample[k] = childsample[k]
                        scope[k] = childsample[k]
            else:
                assert isinstance(prog, Plate)
                platesample = prog.sample(
                    name=childname,
                    scope=scope, 
                    inputs_params=inputs_params.get(childname),
                    active_platedims=active_platedims,
                    all_platedims=all_platedims,
                    groupvarname2Kdim=groupvarname2Kdim,
                    sampler=sampler,
                    reparam=reparam,
                )

                sample[childname] = platesample
                scope[childname] = platesample

        return sample

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
            original_data:Optional[dict[str, Tensor]]):

        if name is not None:
            active_extended_platedims = [*active_extended_platedims, extended_platedims[name]]

        scope = update_scope(scope, inputs_params)
        for childname, childP in self.flat_prog.items():

            childsample = childP.sample_extended(
                sample=sample.get(childname),
                name=childname,
                scope=scope,
                inputs_params=inputs_params.get(childname),
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                active_extended_platedims=active_extended_platedims,
                Ndim=Ndim,
                reparam=reparam,
                original_data=original_data[name] if name is not None else original_data,  # only pass the data for the current plate
            )

            sample[childname] = childsample
            scope = update_scope(scope, {childname: childsample})

        return sample
    
    def predictive_ll(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            original_data:dict[str, Tensor],
            extended_data:dict[str, Tensor]):

        scope = update_scope(scope, inputs_params)

        original_lls, extended_lls = {}, {}

        for childname, childP in self.flat_prog.items():

            child_original_lls, child_extended_lls = childP.predictive_ll(
                sample=sample.get(childname),
                name=childname,
                scope=scope,
                inputs_params=inputs_params.get(childname),
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                original_data=original_data,
                extended_data=extended_data
            )

            scope = update_scope(scope, {childname: sample.get(childname)})

            original_lls = {**original_lls, **child_original_lls}
            extended_lls = {**extended_lls, **child_extended_lls}

        return original_lls, extended_lls
    
    def groupvarname2Kdim(self, K):
        """
        Finds all the Groups/Dists in the program, and creates a 
        K-dimension for each.
        """
        result = {}
        for groupname, v in self.grouped_prog.items():
            if isinstance(v, dict):
                if not datagroup(v):
                    result[groupname] = Dim(f"K_{groupname}", K)
            else:
                assert isinstance(v, Plate)
                result = {**result, **v.groupvarname2Kdim(K)}
        return result

    def all_prog_names(self):
        """
        Returns all plate/group/variable names in the whole program.
        Used to check that all names in the program are unique.
        """
        result = []
        for k, v in self.grouped_prog.items():
            result.append(k)
            if isinstance(v, dict):
                if 2 <= len(v):
                    result = [*result, *v.keys()]
            else:
                assert isinstance(v, Plate)
                result = [*result, *v.all_prog_names()]
        return result

    def varname2groupvarname_dist(self):
        result = {}
        for k, v in self.grouped_prog.items():
            if isinstance(v, dict):
                if not datagroup(v):
                    for gk, gv in v.items():
                        assert isinstance(gv, (Dist, Timeseries))
                        result[gk] = (k, gv)
            else:
                assert isinstance(v, Plate)
                result = {**result, **v.varname2groupvarname_dist()}
        return result

    def varname2groupvarname(self):
        return {varname: groupvarname for (varname, (groupvarname, _)) in self.varname2groupvarname_dist().items()}

    def varname2dist(self):
        return {varname: dist         for (varname, (_, dist))         in self.varname2groupvarname_dist().items()}

    def groupvarname2platenames(self):
        return self._groupvarname2platenames([])

    def _groupvarname2platenames(self, active_platenames:list[str]):
        """
        Returns a dict mapping groupvarname (corresponding to K's) to the names of the active
        plates for that groupvar.

        Used for constructing Js for marginals + posterior sampling
        """
        result = {}
        for name, dgpt in self.grouped_prog.items():
            if isinstance(dgpt, dict):
                result[name] = active_platenames
            else:
                assert isinstance(dgpt, Plate)
                active_platenames = [*active_platenames, name]
                result = {**result, **dgpt._groupvarname2platenames(active_platenames)}
        return result

    def all_platenames(self):
        result = []
        for varname, dgpt in self.flat_prog.items():
            if isinstance(dgpt, Plate):
                result = [*result, *dgpt.all_platenames()]
            else:
                assert isinstance(dgpt, (Dist, Data, Timeseries))
        return result


#Functions to update the scope

def update_scope(scope: dict[str, Tensor], samples_inputs_params:dict):
    assert isinstance(scope, dict)
    assert isinstance(samples_inputs_params, dict)

    scope = {**scope}

    for k, v in samples_inputs_params.items():
        assert k not in scope
        if not isinstance(v, dict):
            assert isinstance(v, Tensor)
            scope[k] = v
    return scope



#### Functions to transform a flat dict to a tree, mirroring the structure of plate.

def empty_tree(plate: Plate):
    assert isinstance(plate, Plate)

    result = {}
    for n, v in plate.flat_prog.items():
        if isinstance(v, Plate):
            result[n] = empty_tree(v)
    return result

def all_platenames(plate: Plate):
    """
    Extracts all platenames from a program
    """
    assert isinstance(plate, Plate)

    result = []
    for n, v in plate.flat_prog.items():
        if isinstance(v, Plate):
            result = [*result, n, *all_platenames(v)]
    return result

def tree_branches(tree:dict):
    result = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            result[k] = v
        else:
            assert isinstance(v, Tensor)
    return result

def tree_values(tree:dict):
    result = {}
    for k, v in tree.items():
        if isinstance(v, Tensor):
            result[k] = v
        else:
            assert isinstance(v, dict)
    return result

def tensordict2tree(plate:Plate, tensor_dict:dict[str, Tensor]):
    root = empty_tree(plate)
    set_all_platenames = set(all_platenames(plate))

    #For each tensor
    for name, tensor in tensor_dict.items():
        current_branch = root

        #Pull out all the plate names
        dimnames = [str(dim) for dim in generic_dims(tensor)]
        platenames = set_all_platenames.intersection(dimnames)

        #Go down tree, until you find the right branch.
        while 0 < len(platenames):
            next_plate = platenames.intersection(tree_branches(current_branch).keys())
            assert 1==len(next_plate)
            next_plate = list(next_plate)[0]

            current_branch = current_branch[next_plate]
            platenames.remove(next_plate)

        current_branch[name] = tensor
    return root

def flatten_tree(tree):
    result = {}
    for k, v in tree.items():
        if isinstance(v, Tensor):
            result[k] = v
        else:
            assert isinstance(v, dict)
            result = {**result, **flatten_tree(v)}
    return result

#def treemap(map_func, reduce_func):
#    def inner(*trees):
#        assert 1 <= len(trees)
#
#        if any(isinstance(tree, dict) in trees):
#            #If one argument is a dict, they're all dicts
#            assert all(isinstance(tree, dict) in trees)
#
#            #If they're all dicts, they have the same keys.
#            keys0 = set(trees[0].keys())
#            assert all(keys0 == set(tree.keys()) for tree in trees[1:])
#
#            #If they're dicts, then you can't apply the function yet,
#            #so keep recursing.
#            result = {}
#            for key in keys0:
#                result[key] = treemap(f, *[tree[key] for tree in trees])
#            return reduce_func(result)
#        else:
#            #If they aren't dicts finally apply the function.
#            return map_func(*trees)
#
#
#def progmap(map_func, reduce_func):
#    def inner(name, trees, active_platedims, **consts):
#        #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
#        #by name=None.
#        if name is not None:
#            new_platedim = all_platedims[name]
#            active_platedims = [*active_platedims, new_platedim]
#
#        plate = trees[0]
#        assert isinstance(plate, Plate)
#
#        result = {}
#        for k, v in plate.prog.items():
#            if isinstance(v, Plate):
#                assert isinstance(tree[k], (Plate, dict))
#                result[k] = inner(k, [tree[k] for tree in trees], active_platedims, **consts)
#            else:
#                result[k] = map_func(k, v, **consts)
#
#        return reduce_func(result, **consts)
