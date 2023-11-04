from typing import Optional, Any

from .Plate import Plate
from .Group import Group
from .dist import Dist
from .tree import DictTree, ListTree

#### Treat everything as a bag of names, and check that the names match for
#### P and Q+data, and don't overlap as appropriate.

def all_names(plate: Plate):
    result = []
    for k, v in plate.prog.items():
        result.append(k)
        if isinstance(v, Group):
            for gk in v.prog:
                result.append(gk)
        elif isinstance(v, Plate):
            result = [*result, *all_names(v)]
    return result

def list_duplicates(xs):
    dups = set()
    xs_so_far = set()
    for x in xs:
        if x in xs_so_far:
            dups.add(x)
        else:
            xs_so_far.add(x)
    return list(dups)

def check_names(P: Plate, Q:Plate, data_names: list[str], bound_names_P: list[str], bound_names_Q: list[str]):
    namesP     = all_names(P)
    namesQdata = [*all_names(Q), *data_names]

    #Check for duplicates in the programs.
    dupsP     = list_duplicates(namesP)
    dupsQdata = list_duplicates(namesQdata)
    if 0 < len(dupsP):
        raise Exception(f"Duplicate names {dupsP} in P")
    if 0 < len(dupsQdata):
        raise Exception(f"Duplicate names {dupsQdata} in Q")

    #Check for mismatches between P and Q+data.
    mismatch_names("Names", namesP, namesQdata)

    #Check there's no overlap between bound names for P and Q
    bound_PQ_overlap = list(set(bound_names_P).intersection(bound_names_Q))
    if 0 < len(bound_PQ_overlap):
        raise Exception(f"Names {bound_PQ_overlap} for inputs/parameters present in both P and Q.  Must be different")

    #Check there's no overlap between bound names and names in the prog
    bound_names = [*bound_names_P, *bound_names_Q]
    set_prog_names = set(namesP) #All the above confirms that prog names in P and Q+data are the same.
    bound_prog_overlap = list(set_prog_names.intersection(bound_names))
    if 0 < len(bound_prog_overlap):
        raise Exception(f"Names {bound_prog_overlap} is both an inputs/parameter and is present in the programs.")

def mismatch_names(prefix:str, namesP: list[str], namesQdata: list[str]):
    #Check for mismatches between P and Q+data.
    inPnotQ = list(set(namesP).difference(namesQdata))
    inQnotP = list(set(namesQdata).difference(namesP))
    if 0 < len(inPnotQ):
        raise Exception(f"{prefix} {inPnotQ} present in P but not Q + data")
    if 0 < len(inQnotP):
        raise Exception(f"{prefix} {inQnotP} present in Q + data, but not in P")



#### Check structure of TreeDict/TreeList corresponds to P
#### Shouldn't arise due to user error, so just asserts.
def check_tree(P, tree):
    for name, dgpt in P.prog.items():
        if isinstance(dgpt, Plate):
            assert name in tree.branches
            check_tree(dgpt, tree.branches[name])



#### Check the structure of the distributions match.

def check_support(name:str, distP:Dist, distQ:Any):
    supportQ = distQ.dist.support 
    supportP = distP.dist.support 
    if supportQ != supportP:
        raise Exception(f"Distributions in P and Q for {nameP} have different support.  For P: {supportP}.  While for Q: {supportQ}")

def check_PQ_group(groupname: str, groupP: Group, groupQ: Group):
    mismatch_names(f"In group {groupname}, there is a mismatch in the keys, with", groupP.prog.keys(), groupQ.prog.keys())
    for varname, distP in groupP.prog.items():
        distQ = groupQ.prog[varname]
        check_support(varname, distP, distQ)


def check_PQ_plate(platename: Optional[str], P: Plate, Q: Plate, data: DictTree):
    """
    Checks that 
    * P and Q have the same Plate/Group structure
    * Distributions in P and Q have the same support
    Doesn't check:
    * Uniqueness of names
    """

    #Check for mismatches between P and Q+data.
    namesP = P.prog.keys()
    assert isinstance(data, DictTree)
    namesQdata = [*Q.prog.keys(), *data.values.keys()]
    mismatch_names(f"In plate {platename}, there is a mismatch in the keys, with", namesP, namesQdata)
    #Now, any name in Q or data must appear in P.

    #Go through the names in data and the names in Q separately.

    #First, names in data.
    #data must correspond to a Dist in P.
    for name in data.values.keys():
        distP = P.prog[name]
        if not isinstance(distP, Dist):
            raise Exception(f"{name} in appears in Plate {platename} as data, so the corresponding {name} in P should be a distribution over a single random variable.  But actually {name} in P is something else: {type(distP)}")

    #Now check names in Q 
    for name, dgpt_Q in P.prog.items():
        if isinstance(dgpt_Q, Dist):
            distQ = dgpt_Q
            distP = P.prog[name]
            if not isinstance(distP, Dist):
                raise Exception(f"{name} in Q is a Dist, so it should also be a Group in P, but actually its a {type(distP)}.")
            check_support(name, distP, distQ)
        elif isinstance(dgpt_Q, Group):
            groupQ = dgpt_Q
            groupP = P.prog[name]
            if not isinstance(groupP, Group):
                raise Exception(f"{name} in Q is a Group, so it should also be a Group in P, but actually its a {type(groupP)}.")
            check_PQ_group(name, groupP, groupQ)
        else:
            assert isinstance(dgpt_Q, Plate)
            plateQ = dgpt_Q
            plateP = P.prog[name]
            if not isinstance(plateP, Plate):
                raise Exception(f"{name} in Q is a Plate, so it should also be a Plate in P, but actually its a {type(plateP)}.")
            check_PQ_plate(name, plateP, plateQ, data.branches[name])
