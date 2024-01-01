import torch as t
import torch.nn as nn
from typing import Union

from .Plate import Plate, tensordict2tree, flatten_tree
from .BoundPlate import BoundPlate, named2torchdim_flat2tree
from .Sampler import Sampler
from .utils import *
from .checking import check_PQ_plate, check_inputs_params, mismatch_names
from .logpq import logPQ_plate
from .Sampler import PermutationSampler
from .Stores import BufferStore

from .Sample import Sample

PBP = Union[Plate, BoundPlate]


class Problem(nn.Module):
    """
    Combines the prior, approximate posterior and data.

    Arguments:
        P (BoundPlate):
            The prior, as a BoundPlate.
        Q (BoundPlate):
            The approximate posterior, as a BoundPlate.
        data (dict[str, (named) torch.Tensor]):
            A dict mapping the data variable name to a named ``torch.Tensor``, where the names correspond to plates.
    """
    def __init__(self, P:BoundPlate, Q:BoundPlate, data: dict[str, t.Tensor]):
        super().__init__()

        if (not isinstance(P, BoundPlate)) or (not isinstance(Q, BoundPlate)):
            raise Exception("P and Q must be `BoundPlate`s, not e.g. `Plate`s.  You can convert just using `bound_plate_P = BoundPlate(plate_P)` if it doesn't have any inputs or parameters")

        #A tensor that e.g. moves to GPU when we call `problem.to(device='cuda')`.
        self.register_buffer("_device_tensor", t.zeros(()))

        self.P = P
        self.Q = Q

        if P.all_platesizes != Q.all_platesizes:
            raise Exception(f"all_platesizes does not match between P and Q.  In P it is {P.all_platesizes}, while in Q it is {Q.all_platesizes}")
        
        self.all_platedims = {name: Dim(name, size) for name, size in P.all_platesizes.items()}
        #Put data in a BufferDict so that it is registered properly, and moves to device as requested.
        self._data = BufferStore(data)


        #Check names in P matches those in Q+data, and there are no duplicates.
        #Check the structure of P matches that of Q.
        check_PQ_plate(None, P.plate, Q.plate, self.data)
        check_inputs_params(P, Q)

    @property
    def data(self):
        """
        Converts data to a torchdim tree
        """
        return tensordict2tree(self.P.plate, named2dim_dict(self._data.to_dict(), self.all_platedims))

    @property
    def device(self):
        return self._device_tensor.device

    def check_device(self):
        if not (self.device == self.P.device and self.device == self.Q.device):
            raise Exception("Device issue: Problem, P and/or Q aren't all on the same device.  The easiest way to make sure everything works is to call e.g. problem.to('cuda'), rather than e.g. P.to('cuda').")

    def sample(self, K: int, reparam:bool=True, sampler:Sampler=PermutationSampler):
        """
        alan.sample(K:int, reparam=True, sampler=PermutationSampler)

        Draws K samples for each latent variable from the approximate posterior, and returns the result as a Sample class.

        Arguments:
            K (int):
                Number of samples to draw for each latent variable.

        Keyword Arguments:
            reparam (bool):
                Whether to use the reparameterisation trick to differentiate through the sample generation process.  Note that whatever you want to do downstream, it is always safe to leave the default ``reparam=True``.  That's because once you've drawn a reparameterised sample, you can always detach it.  The only reason this parameter is still included is that if you don't need reparameterisation, it will save memory slightly to set ``reparam=False``.
            sampler (Sampler):
                When your approximate posterior is not factorised, there is a slightly subtle choice of which of the K samples of parent latent variables to depend on.  We currently have two choices.  ``alan.CategoricalSampler``, in which each downstream sample choose one parent particle at random.  However, that doesn't work so well, because of the "particle degeneracy" problem also encountered in particle filters.  Specifically, one upstream particle may have zero, or multiple downstream children, which reduces diversity.  To circumvent this problem, we also have ``alan.PermutationSampler``, in which each parent particle has exactly one child.  The default ``alan.PermutationSampler`` seems to work fine, and we don't really see a need for users to ever change the default.
        """
        self.check_device()

        sample, groupvarname2Kdim = self.Q._sample(K, reparam, sampler, self.all_platedims)

        return Sample(
            problem=self,
            sample=sample,
            groupvarname2Kdim=groupvarname2Kdim,
            sampler=sampler,
            reparam=reparam,
        )

    def inputs_params(self):
        flat_named = {
            **self.P.inputs_params_flat_named(), 
            **self.Q.inputs_params_flat_named()
        }
        return named2torchdim_flat2tree(flat_named, self.all_platedims, self.P.plate)
