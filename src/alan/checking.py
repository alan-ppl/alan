from typing import Optional, Any, Union

from .Plate import Plate, tree_branches, tree_values
from .BoundPlate import BoundPlate

from .Group import Group
from .dist import Dist
from .Data import Data
from .Timeseries import Timeseries



#### Check the structure of the distributions match.

def check_inputs_params(P:BoundPlate, Q:BoundPlate):
    assert isinstance(P, BoundPlate)
    assert isinstance(Q, BoundPlate)

    inputs_params_P = P.inputs_params_flat_named()
    inputs_params_Q = Q.inputs_params_flat_named()

    overlap = set(inputs_params_P.keys()).intersection(inputs_params_Q.keys())

    for k in overlap:
        if not (inputs_params_P[k] == inputs_params_Q[k]).rename(None).all():
            raise Exception(f"Input / parameter names must be different in P and Q (or they must refer to the same input/parameter).  However, {k} refers to different inputs/parameters in P and Q.  Note that this can happen if you use OptParam / QEMParam for the same parameters in P and Q.  In that case, you should use the explicit `name` kwarg on OptParam/QEMParam.  e.g. `OptParam(1., name='a_loc_P')`")


def check_support(name:str, distP:Dist, distQ:Dist):
    assert isinstance(distP, Dist)
    assert isinstance(distQ, Dist)

    supportQ = distQ.dist.support 
    supportP = distP.dist.support 
    if supportQ != supportP:
        raise Exception(f"Distributions in P and Q for {nameP} have different support.  For P: {supportP}.  While for Q: {supportQ}")

def mismatch_names(A: list[str], B: list[str], prefix="", AnotB_msg="", BnotA_msg=""):
    #Check for mismatches between two lists of names
    inAnotB = list(set(A).difference(B))
    inBnotA = list(set(B).difference(A))
    if 0 < len(inAnotB):
        raise Exception(f"{prefix} {inAnotB} {AnotB_msg}.")
    if 0 < len(inBnotA):
        raise Exception(f"{prefix} {inBnotA} {BnotA_msg}.")
    
def mismatch_PG_varnames(P:list[str], Q:list[str], area:str):
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
    namesP = P.flat_prog.keys()
    namesQ = Q.flat_prog.keys()
    mismatch_PG_varnames(namesP, namesQ, area=f"plate {platename}")

    #Check data names match between Q and data.
    data_names_in_Q = [k for (k, v) in Q.flat_prog.items() if isinstance(v, Data)]
    data_names      = tree_values(data).keys()
    mismatch_names(
        data_names_in_Q, data_names,
        prefix=f"There is a mismatch in the variable names in the data dict given as an argument to Problem ({list(data_names)}), and those given in Q using e.g. `varname=Data()` ({data_names_in_Q}). Specifically, there is an issue in plate {platename}, with",
        AnotB_msg=f"present as `varname=Data()` in Q but not in the data dict provided to Problem",
        BnotA_msg=f"present in the data dict provided to Problem, but not given as `=Data()` in Q",
    )

    #Now check names in Q 
    for name, dgpt_P in P.flat_prog.items():
        if isinstance(dgpt_P, Dist):
            distP = dgpt_P
            distQ = Q.flat_prog[name]
            if not isinstance(distQ, (Dist, Data)):
                raise Exception(f"{name} in P is a Dist, so {name} in Q should be a Data/Dist, but actually its a {type(distQ)}.")
            if isinstance(distQ, Dist):
                check_support(name, distP, distQ)

        elif isinstance(dgpt_P, Timeseries):
            timeseries_P = dgpt_P
            timeseries_dist_Q = Q.flat_prog[name]
            if not isinstance(timeseries_dist_Q, (Dist, Timeseries, Data)):
                raise Exception(f"{name} in P is a Timeseries, so {name} in Q should be a Timeseries or a Dist, but actually its a {type(groupQ)}.")
            dist_Q = timeseries_dist_Q.trans if isinstance(timeseries_dist_Q, Timeseries) else timeseries_dist_Q
            check_support(name, timeseries_P.trans, dist_Q)

        elif isinstance(dgpt_P, Group):
            groupP = dgpt_P
            groupQ = Q.flat_prog[name]
            if not isinstance(groupQ, Group):
                raise Exception(f"{name} in P is a Group, so {name} in Q should also be a Group, but actually its a {type(groupQ)}.")

        elif isinstance(dgpt_P, Plate):
            plateP = dgpt_P
            plateQ = Q.flat_prog[name]
            if not isinstance(plateQ, Plate):
                raise Exception(f"{name} in P is a Plate, so {name} in Q should also be a Plate, but actually its a {type(plateQ)}.")

            #Recurse
            check_PQ_plate(name, plateP, plateQ, data[name])
        elif isisntance(dgpt_P, Data):
            raise Exception(f"{name} in P is Data.  But we can't have Data in P.")
        else:
            raise Exception(f"{name} is an unrecognised type (should be Plate, Group, Dist or Data (but can only be data in Q))")
