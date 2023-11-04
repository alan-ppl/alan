from functorch.dim import Dim

from .utils import generic_dims, Tensor
from .Plate import Plate


class Tree:
    def __init__(self, values):
        self.values = values
        self.branches = {}

    def get(self, idx):
        return self.branches.get(idx)

class DictTree(Tree):
    def __init__(self):
        super().__init__({})

class ListTree(Tree):
    def __init__(self, branches:dict[str, Tree]):
        super().__init__([])

def empty_ListTree_from_PQ(pq: Plate):
    return empty_tree_from_PQ(ListTree, pq)

def empty_DictTree_from_PQ(pq: Plate):
    return empty_tree_from_PQ(DictTree, pq)

def empty_tree_from_PQ(treetype, pq:Plate):
    tree = treetype()
    for n, v in pq.prog.items():
        if isinstance(v, Plate):
            tree.branches[n] = empty_tree_from_PQ(treetype, v)
    return tree

def all_platenames(pq: Plate):
    """
    Extracts all platenames from a program
    """
    result = []
    for n, v in pq.prog.items():
        if isinstance(v, Plate):
            result = [*result, n, *all_platenames(v)]
    return result

def tensorlist2tree(pq:Plate, tensor_list:list[Tensor]):
    root = empty_ListTree_from_PQ(pq)
    set_all_platenames = set(all_platenames(pq))

    #For each tensor
    for tensor in tensor_list:
        current_branch = root

        #Pull out all the plate names
        dimnames = [str(dim) for dim in generic_dims(tensor)]
        platenames = set_all_platenames.intersection(dimnames)

        #Go down tree, until you find the right branch.
        while 0 < len(platenames):
            next_plate = platenames.intersection(current_branch.branches.keys())
            assert 1==len(next_plate)
            next_plate = list(next_plate)[0]

            current_branch = current_branch.branches[next_plate]
            platenames.remove(next_plate)

        current_branch.values.append(tensor)
    return root

def tensordict2tree(pq:Plate, tensor_dict:dict[Tensor]):
    root = empty_DictTree_from_PQ(pq)
    set_all_platenames = set(all_platenames(pq))

    #For each tensor
    for name, tensor in tensor_dict.items():
        current_branch = root

        #Pull out all the plate names
        dimnames = [str(dim) for dim in generic_dims(tensor)]
        platenames = set_all_platenames.intersection(dimnames)

        #Go down tree, until you find the right branch.
        while 0 < len(platenames):
            next_plate = platenames.intersection(current_branch.branches.keys())
            assert 1==len(next_plate)
            next_plate = list(next_plate)[0]

            current_branch = current_branch.branches[next_plate]
            platenames.remove(next_plate)

        current_branch.values[name] = tensor
    return root
