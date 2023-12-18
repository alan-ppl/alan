import torch as t
from typing import Union

from .Plate import Plate, tensordict2tree, flatten_tree
from .BoundPlate import BoundPlate
from .SamplingType import SamplingType
from .utils import *
from .checking import check_PQ_plate, check_inputs_params, mismatch_names
from .logpq import logPQ_plate

from .Sample import Sample

PBP = Union[Plate, BoundPlate]



class Problem():
    def __init__(self, P: BoundPlate, Q: BoundPlate, all_platesizes: dict[str, int], data: dict[str, t.Tensor]):
        assert isinstance(P, BoundPlate)
        assert isinstance(Q, BoundPlate)

        self.P = P
        self.Q = Q
        self.all_platedims = {name: Dim(name, size) for name, size in all_platesizes.items()}
        self.data = tensordict2tree(P.plate, named2dim_dict(data, self.all_platedims))

        #Check names in P matches those in Q+data, and there are no duplicates.
        #Check the structure of P matches that of Q.
        check_PQ_plate(None, P.plate, Q.plate, self.data)
        check_inputs_params(P, Q)

        P.check_deps(self.all_platedims)
        Q.check_deps(self.all_platedims)

    def sample(self, K: int, reparam:bool, sampling_type:SamplingType):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        groupvarname2Kdim = self.Q.plate.groupvarname2Kdim(K)

        sample = self.Q.plate.sample(
            name=None,
            scope={},
            inputs_params=self.Q.inputs_params(self.all_platedims),
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            reparam=reparam,
        )

        result = Sample(
            problem=self,
            sample=sample,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            split=None,
        )
    
        return result

    def groupvarname2parent_groupvarnames(self):
        """
        Returns a dictionary giving the dependencies for each groupvar (i.e. variables
        outside groups and groups).  Only includes those in Q, so should correspond to
        dependencies for Ks.
        """
        varname2groupvarname = self.Q.varname2groupvarname()
        groupvarnames_inQ = self.Q.groupvarnames()


        #Has all keys, including keys for data; also maps to varname, not groupvarname
        result = {}
        for gvn, parents in self.P.groupvarname2parents().items():
            #Only include groups or variables in Q.
            if gvn in groupvarnames_inQ:
                modified_parents = []
                for parent in parents:
                    #Only include parents in Q.
                    if parent in varname2groupvarname:
                        modified_parents.append(varname2groupvarname[parent])
                result[gvn] = modified_parents
        return result
