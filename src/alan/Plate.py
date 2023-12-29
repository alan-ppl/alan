import torch as t
from typing import Optional

from functorch.dim import Dim

from .utils import *
from .Sampler import Sampler
from .dist import Dist
from .Group import Group
from .Data import Data
from .dist import Dist, _Dist




class Plate():
    def __init__(self, **kwargs):
        self.prog = {}

        #Finalize the distributions by passing in the varname, and check types
        for varname, dgpt in kwargs.items():
            if isinstance(dgpt, _Dist):
                dgpt = dgpt.finalize(varname)
            assert isinstance(dgpt, (Dist, Group, Data, Plate))
            self.prog[varname] = dgpt


        #Error checking: plate/variable/group names aren't reserved
        all_prog_names = self.all_prog_names()
        for name in all_prog_names:
            check_name(name)

        #Error checking: no duplicate names.
        dup_names = list_duplicates(all_prog_names)
        if 0 != len(dup_names):
            raise Exception(f"Plate has duplicate names {dup_names}.")

    def finalize_init(self, groupvarname):
        for groupvarname, dgpt in self.prog.items():
            if isinstance(dgpt, (Dist, Plate, Group)):
                dgpt.finalize_init(groupvarname)
            else:
                assert isinstance(dgpt, Data)

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

        scope = update_scope_inputs_params(scope, inputs_params)
        sample = {}
        
        for childname, dgpt in self.prog.items():
            if not isinstance(dgpt, Data):
                childsample = dgpt.sample(
                    name=childname,
                    scope=scope, 
                    inputs_params=inputs_params.get(childname),
                    active_platedims=active_platedims,
                    all_platedims=all_platedims,
                    groupvarname2Kdim=groupvarname2Kdim,
                    sampler=sampler,
                    reparam=reparam,
                )

                sample[childname] = childsample
                scope = update_scope_sample(scope, childname, dgpt, childsample)

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

        scope = update_scope_inputs_params(scope, inputs_params)

        for childname, childP in self.prog.items():

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
            scope = update_scope_sample(scope, childname, childP, childsample)

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

        scope = update_scope_inputs_params(scope, inputs_params)

        original_lls, extended_lls = {}, {}

        for childname, childP in self.prog.items():

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

            scope = update_scope_sample(scope, childname, childP, sample.get(childname))

            original_lls = {**original_lls, **child_original_lls}
            extended_lls = {**extended_lls, **child_extended_lls}

        return original_lls, extended_lls
    
    def groupvarname2Kdim(self, K):
        """
        Finds all the Groups/Dists in the program, and creates a 
        K-dimension for each.
        """
        result = {}
        for childname, childP in self.prog.items():
            if isinstance(childP, (Dist, Group)):
                result[childname] = Dim(f"K_{childname}", K)
            elif isinstance(childP, Plate):
                assert isinstance(childP, Plate)
                result = {**result, **childP.groupvarname2Kdim(K)}
        return result

    def all_prog_names(self):
        """
        Returns all plate/group/variable names in the whole program.
        Used to check that all names in the program are unique.
        """
        result = []
        for k, v in self.prog.items():
            result.append(k)
            if isinstance(v, (Plate, Group)):
                result = [*result, *v.all_prog_names()]
            else:
                assert isinstance(v, (Dist, Data))
        return result

    def varname2groupvarname_dist(self):
        result = {}
        for k, v in self.prog.items():
            if isinstance(v, Dist):
                varname = k
                groupvarname = k
                dist = v
                result[varname] = (groupvarname, dist)
            elif isinstance(v, Group):
                for gk, gv in v.prog.items():
                    varname = gk
                    groupvarname = k
                    dist = gv
                    assert isinstance(dist, Dist)
                    result[varname] = (groupvarname, dist)
            elif isinstance(v, Plate):
                result = {**result, **v.varname2groupvarname_dist()}
            else:
                assert isinstance(v, Data)
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
        for name, dgpt in self.prog.items():
            if isinstance(dgpt, (Dist, Group)):
                result[name] = active_platenames
            elif isinstance(dgpt, Plate):
                active_platenames = [*active_platenames, name]
                result = {**result, **dgpt._groupvarname2platenames(active_platenames)}
            else:
                assert isinstance(dgpt, Data)
        return result

    #def varname2active_platedimnames(self):
    #    groupvarname2active_platedimnames = self.groupvarname2active_platedimnames()

    #    result = {}
    #    for (varname, groupvarname) in self.varname2groupvarname().items():
    #        result[varname] = groupvarname2active_platedimnames[groupvarname]
    #    return result


#Functions to update the scope

def update_scope_sample(scope: dict[str, Tensor], name:str, dgpt, sample):
    return update_scope_samples(scope, {name:dgpt}, {name: sample})

def update_scope_samples(scope: dict[str, Tensor], Q_prog:dict, samples:dict):
    scope = {**scope}

    for childname, childQ in Q_prog.items():
        if isinstance(childQ, Data):
            assert childname not in samples
        else:
            sample = samples[childname]

            if isinstance(childQ, Dist):
                assert isinstance(sample, Tensor)
                scope[childname] = sample
            elif isinstance(childQ, Group):
                assert isinstance(sample, dict)
                for gn, gs in sample.items():
                    assert isinstance(gs, Tensor)
                    scope[gn] = gs
            else:
                assert isinstance(childQ, Plate)
    return scope

def update_scope_inputs_params(scope:dict[str, Tensor], inputs_params:dict):
    scope = {**scope}
    for n, v in inputs_params.items():
        if isinstance(v, Tensor):
            scope[n] = v
        else:
            assert isinstance(v, dict)
    return scope

def update_scope(scope:dict[str, Tensor], P:Plate, sample:dict, inputs_params:dict):
    scope = update_scope_inputs_params(scope, inputs_params)
    scope = update_scope_samples(scope, P.prog, sample)
    return scope



#### Functions to transform a flat dict to a tree, mirroring the structure of plate.

def empty_tree(plate: Plate):
    assert isinstance(plate, Plate)

    result = {}
    for n, v in plate.prog.items():
        if isinstance(v, Plate):
            result[n] = empty_tree(v)
    return result

def all_platenames(plate: Plate):
    """
    Extracts all platenames from a program
    """
    assert isinstance(plate, Plate)

    result = []
    for n, v in plate.prog.items():
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

def treemap(f, *trees):
    assert 1 <= len(trees)

    if any(isinstance(tree, dict) in trees):
        #If one argument is a dict, they're all dicts
        assert all(isinstance(tree, dict) in trees)

        #If they're all dicts, they have the same keys.
        keys0 = set(trees[0].keys())
        assert all(keys0 == set(tree.keys()) for tree in trees[1:])

        #If they're dicts, then you can't apply the function yet,
        #so keep recursing.
        result = {}
        for key in keys0:
            result[key] = treemap(f, *[tree[key] for tree in trees])

        return result
    else:
        #If they aren't dicts finally apply the function.
        return f(*trees)

def flatten_tree(tree):
    result = {}
    for k, v in tree.items():
        if isinstance(v, Tensor):
            result[k] = v
        else:
            assert isinstance(v, dict)
            result = {**result, **flatten_tree(v)}
    return result


