from typing import Optional

from functorch.dim import Dim

from .utils import *
from .SamplingType import SamplingType
from .dist import Dist
from .Group import Group
from .utils import *



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
            reparam:bool):

        if name is not None:
            active_platedims = [*active_platedims, all_platedims[name]]

        scope = update_scope_inputs_params(scope, inputs_params)
        sample = {}

        for childname, childP in self.prog.items():

            childsample = childP.sample(
                name=childname,
                scope=scope, 
                inputs_params=inputs_params.get(childname),
                active_platedims=active_platedims,
                all_platedims=all_platedims,
                groupvarname2Kdim=groupvarname2Kdim,
                sampling_type=sampling_type,
                reparam=reparam,
            )

            sample[childname] = childsample
            scope = update_scope_sample(scope, childname, childP, childsample)

        return sample

    def groupvarname2Kdim(self, K):
        """
        Finds all the Groups/Dists in the program, and creates a 
        K-dimension for each.
        """
        result = {}
        for childname, childP in self.prog.items():
            if isinstance(childP, (Dist, Group)):
                result[childname] = Dim(f"K_{childname}", K)
            else:
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
                assert isinstance(v, Dist)
        return result

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
            else:
                assert isinstance(v, Plate)
                result = {**result, **v.varname2groupvarname()}
        return result
                
    def inputs_params(self, all_platedims:dict[str, Dim]):
        """
        Returns all the inputs/params used in the program.  Empty
        as these are only defined for BoundPlate.
        """
        return empty_tree(self)

    def groupvarname2active_platedimnames(self, active_platedimnames=None):
        """
        Returns a dict mapping groupvarname (corresponding to K's) to the names of the active
        plates for that groupvar.

        Used for constructing Js for marginals + posterior sampling
        """
        if active_platedimnames is None:
            active_platedimnames = []

        result = {}
        for name, dgpt in self.prog.items():
            if isinstance(dgpt, (Dist, Group)):
                result[name] = active_platedimnames
            else:
                assert isinstance(dgpt, Plate)
                active_platedimnames = [*active_platedimnames, name]
                result = {**result, **dgpt.groupvarname2active_platedimnames(active_platedimnames)}
        return result

    def groupvarname2parents(self):
        """
        Returns a dict mapping groupvarname (corresponding to K's) to the names of the
        arguments. Arguments could be variables, inputs or parameters.

        Used for constructing Js for posterior sampling
        """
        result = {}
        for vargroupname, dgpt in self.prog.items():
            if isinstance(dgpt, (Dist, Group)):
                result[vargroupname] = dgpt.all_args
            else:
                assert isinstance(dgpt, Plate)
                result = {**result, **dgpt.groupvarname2parents()}
        return result


#### Functions to update the scope.

def update_scope_sample(scope: dict[str, Tensor], name:str, dgpt, sample):
    scope = {**scope}
    if isinstance(dgpt, Dist):
        assert isinstance(sample, Tensor)
        scope[name] = sample
    elif isinstance(dgpt, Group):
        assert isinstance(sample, dict)
        for gn, gs in sample.items():
            assert isinstance(gs, Tensor)
            scope[gn] = gs
    else:
        assert isinstance(dgpt, Plate)
    return scope


def update_scope_inputs_params(scope:dict[str, Tensor], inputs_params:dict):
    scope = {**scope}
    for n, v in inputs_params.items():
        if isinstance(v, Tensor):
            scope[n] = v
        else:
            assert isinstance(v, dict)
    return scope




#### Functions to transform a flat dict to a tree, mirroring the structure of plate.

def empty_tree(plate: Plate):
    result = {}
    for n, v in plate.prog.items():
        if isinstance(v, Plate):
            result[n] = empty_tree(v)
    return result

def all_platenames(plate: Plate):
    """
    Extracts all platenames from a program
    """
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


