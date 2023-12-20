import torch as t
from typing import Optional

from functorch.dim import Dim

from .utils import *
from .SamplingType import SamplingType, IndependentSample
from .dist import Dist
from .Group import Group
from .Data import Data




class Plate():
    def __init__(self, **kwargs):
        self.prog = kwargs

        #Error checking: plate/variable/group names aren't reserved
        all_prog_names = self.all_prog_names()
        for name in all_prog_names:
            check_name(name)

        #Error checking: no duplicate names.
        dup_names = list_duplicates(all_prog_names)
        if 0 != len(dup_names):
            raise Exception(f"Plate has duplicate names {dup_names}.")


    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            inputs_params: dict,
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampling_type:SamplingType,
            reparam:bool,
            device:t.device,
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
                    sampling_type=sampling_type,
                    reparam=reparam,
                    device=device,
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
            active_original_platedims:list[Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:Optional[dict[str, Tensor]],
            extended_data:Optional[dict[str, Tensor]]):

        # NOTE: I think we might be able to get rid of active_original_platedims and active_extended_platedims
        #  in this function (and the corresponding functions in Group and Dist) as they can be inferred from
        #  the samples/data and the dicts original_platedims and extended_platedims.
        #  At the moment, only active_extended_platedims is used: in the Dist function when calculating the
        #  logprobs of extended data, but there might be a smarter way of providing extended_data that 
        #  circumvents this need.
        if name is not None:
            active_original_platedims = [*active_original_platedims, original_platedims[name]]
            active_extended_platedims = [*active_extended_platedims, extended_platedims[name]]

        scope = update_scope_inputs_params(scope, inputs_params)

        original_ll = {}
        extended_ll = {}

        for childname, childP in self.prog.items():

            childsample, child_original_ll, child_extended_ll = childP.sample_extended(
                sample=sample.get(childname),
                name=childname,
                scope=scope,
                inputs_params=inputs_params.get(childname),
                original_platedims=original_platedims,
                extended_platedims=extended_platedims,
                active_original_platedims=active_original_platedims,
                active_extended_platedims=active_extended_platedims,
                Ndim=Ndim,
                reparam=reparam,
                original_data=original_data[name] if name is not None else original_data,  # only pass the data for the current plate
                extended_data=extended_data # pass all extended data, this is a flat dict whereas original_data has a tree structure
            )

            sample[childname] = childsample
            scope = update_scope_sample(scope, childname, childP, childsample)

            original_ll = {**original_ll, **child_original_ll}
            extended_ll = {**extended_ll, **child_extended_ll}


        return sample, original_ll, extended_ll


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

#    def groupvarname2varnames(self):
#        result = {}
#        for k, v in self.prog.items():
#            if isinstance(v, Dist):
#                result[k] = (k,)
#            elif isinstance(v, Group):
#                result[k] = tuple(v.prog.keys())
#            elif isinstance(v, Plate):
#                result = {**result, **v.groupvarname2varnames()}
#            else:
#                assert isinstance(v, Data)
#        return result
#
    def varname2groupvarname(self):
        """
        Returns a dictionary mapping the variable name to the groupvarname
        i.e. for standard random variables, this is an identity mapping, but
        for variables in a group, it takes the name of the variable, and
        returns the name of the group.
        """
        result = {}
        for k, v in self.prog.items():
            if isinstance(v, Dist):
                result[k] = k
            elif isinstance(v, Group):
                for gk, gv in v.prog.items():
                    result[gk] = k
            elif isinstance(v, Plate):
                result = {**result, **v.varname2groupvarname()}
            else:
                assert isinstance(v, Data)
        return result
#
#    def varnames(self):
#        """
#        Returns a list of varnames in the Plate.
#        """
#        result = []
#        for k, v in self.prog.items():
#            if isinstance(v, Dist):
#                result.append(k)
#            elif isinstance(v, Group):
#                result = [*result, *v.prog.keys()]
#            elif isinstance(v, Plate):
#                result = [*result, *v.varnames()]
#            else:
#                assert isinstance(v, Data)
#        return result
#
#    def groupvarnames(self):
#        """
#        Returns a list of groupvarnames in the Plate (i.e. includes group
#        names but not variables in a group).  Corresponds to K-dimensions.
#        """
#        result = []
#        for k, v in self.prog.items():
#            if isinstance(v, (Group, Dist)):
#                result.append(k)
#            elif isinstance(v, Plate):
#                result = [*result, *v.groupvarnames()]
#            else:
#                assert isinstance(v, Data)
#        return result
#                
    def groupvarname2active_platedimnames(self, active_platedimnames:list[str]):
        """
        Returns a dict mapping groupvarname (corresponding to K's) to the names of the active
        plates for that groupvar.

        Used for constructing Js for marginals + posterior sampling
        """
        result = {}
        for name, dgpt in self.prog.items():
            if isinstance(dgpt, (Dist, Group)):
                result[name] = active_platedimnames
            elif isinstance(dgpt, Plate):
                active_platedimnames = [*active_platedimnames, name]
                result = {**result, **dgpt.groupvarname2active_platedimnames(active_platedimnames)}
            else:
                assert isinstance(dgpt, Data)
        return result
#
#    def groupvarname2parents(self):
#        """
#        Returns a dict mapping groupvarname (corresponding to K's) to the names of the
#        arguments. Arguments could be variables, inputs or parameters.
#
#        Used for constructing Js for posterior sampling
#        """
#        result = {}
#        for vargroupname, dgpt in self.prog.items():
#            if isinstance(dgpt, (Dist, Group)):
#                result[vargroupname] = dgpt.all_args
#            elif isinstance(dgpt, Plate):
#                result = {**result, **dgpt.groupvarname2parents()}
#            else:
#                assert isinstance(v, Data)
#        return result


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


