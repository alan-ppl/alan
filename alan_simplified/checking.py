from typing import Optional, Any, Union

from .Plate import Plate, tree_branches, tree_values
from .BoundPlate import BoundPlate

from .Group import Group
from .dist import Dist
from .Data import Data

#### Check the structure of the distributions match.

def check_inputs_params(P:BoundPlate, Q:BoundPlate):
    assert isinstance(P, BoundPlate)
    assert isinstance(Q, BoundPlate)

    inputs_params_P = P.inputs_params_flat_named()
    inputs_params_Q = Q.inputs_params_flat_named()

    overlap = set(inputs_params_P.keys()).intersection(inputs_params_Q.keys())

    for k in overlap:
        if inputs_params_P[k] is not inputs_params_Q[k]:
            raise Exception(f"There is an input or parameter that is shared between P and Q, and isn't the same for both")


def check_support(name:str, distP:Dist, distQ:Dist):
    assert isinstance(distP, Dist)
    assert isinstance(distQ, Dist)

    supportQ = distQ.dist.support 
    supportP = distP.dist.support 
    if supportQ != supportP:
        raise Exception(f"Distributions in P and Q for {nameP} have different support.  For P: {supportP}.  While for Q: {supportQ}")

def check_PQ_group(groupname: str, groupP: Group, groupQ: Group):
    mismatch_pg_varnames(groupP.prog.keys(), groupQ.prog.keys(), area=f"group {groupname}")

    for varname, distP in groupP.prog.items():
        distQ = groupQ.prog[varname]
        if isinstance(distQ, Data):
            raise Exception("Cannot have data inside a Group")

        check_support(varname, distP, distQ)

def mismatch_names(A: list[str], B: list[str], prefix="", AnotB_msg="", BnotA_msg=""):
    #Check for mismatches between two lists of names
    inAnotB = list(set(A).difference(B))
    inBnotA = list(set(B).difference(A))
    if 0 < len(inAnotB):
        raise Exception(f"{prefix} {inAnotB} {AnotB_msg}.")
    if 0 < len(inBnotA):
        raise Exception(f"{prefix} {inBnotA} {BnotA_msg}.")
    
def mismatch_pg_varnames(P:list[str], Q:list[str], area:str):
    mismatch_names(
        P, Q,
        prefix=f"In {area}, there is a mismatch in the variable names, with",
        AnotB_msg="present in P but not Q",
        BnotA_msg="present in Q but not P",
    )


def check_PQ_plate(platename: Optional[str], P: Plate, Q: Plate, data: dict):
    """
    Checks that 
    * P and Q have the same Plate/Group structure
    * Distributions in P and Q have the same support
    Doesn't check:
    * Uniqueness of names
    """

    #Check for mismatches in the varnames between.
    namesP = P.prog.keys()
    namesQ = Q.prog.keys()
    mismatch_pg_varnames(namesP, namesQ, area=f"plate {platename}")

    #Check data names match between Q and data.
    data_names_in_Q = [k for (k, v) in Q.prog.items() if isinstance(v, Data)]
    data_names      = tree_values(data).keys()
    breakpoint()
    mismatch_names(
        data_names_in_Q, data_names,
        prefix=f"There is a mismatch in the names for data in Q and in the provided data in plate {platename}, with",
        AnotB_msg=f"present in Q but not in data",
        BnotA_msg=f"present in data but not in Q",
    )

    #Now check names in Q 
    for name, dgpt_P in P.prog.items():
        if isinstance(dgpt_P, Dist):
            distP = dgpt_P
            distQ = Q.prog[name]
            if not isinstance(distQ, (Dist, Data)):
                raise Exception(f"{name} in P is a Dist, so {name} in Q should be a Data/Dist, but actually its a {type(distQ)}.")
            if isinstance(distQ, Dist):
                check_support(name, distP, distQ)

        elif isinstance(dgpt_P, Group):
            groupP = dgpt_P
            groupQ = Q.prog[name]
            if not isinstance(groupQ, Group):
                raise Exception(f"{name} in P is a Group, so {name} in Q should also be a Group, but actually its a {type(groupQ)}.")
            #Recurse
            check_PQ_group(name, groupP, groupQ)

        elif isinstance(dgpt_P, Plate):
            plateP = dgpt_P
            plateQ = Q.prog[name]
            if not isinstance(plateQ, Plate):
                raise Exception(f"{name} in P is a Plate, so {name} in Q should also be a Plate, but actually its a {type(plateQ)}.")

            #Recurse
            check_PQ_plate(name, plateP, plateQ, data[name])
        elif isisntance(dgpt_P, Data):
            raise Exception(f"{name} in P is Data.  But we can't have Data in P.")
        else:
            raise Exception(f"{name} is an unrecognised type (should be Plate, Group, Dist or Data (but can only be data in Q))")
