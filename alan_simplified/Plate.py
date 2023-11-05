from typing import Optional

from functorch.dim import Dim

from .utils import *
from .SamplingType import SamplingType
from .dist import Dist
from .Group import Group
from .utils import *


class PlateTimeseries():
    pass

class Plate(PlateTimeseries):
    def __init__(self, **kwargs):
        self.prog = kwargs

        #Error checking: plate/variable/group names aren't reserved
        all_names = self.all_names()
        for name in all_names:
            check_name(name)

        #Error checking: no duplicate names.
        dup_names = list_duplicates(all_names)
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

        parent_scope = scope
        scope = update_scope(scope, inputs_params)
        sample = {}

        for childname, childP in self.prog.items():

            childsample, scope = childP.sample(
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

        return sample, parent_scope

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
                result = [*result, *all_names(v)]
            else:
                assert isinstance(v, Dist)
        return result
                
    def inputs_params_named(self):
        """
        Returns all the inputs/params used in the program.  Empty
        as these are only defined for BoundPlate.
        """
        return empty_tree(self)


        

class Timeseries(PlateTimeseries):
    pass



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

