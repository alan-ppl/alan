######## ! WARNING ! ########
# This is a copy of torch.unravel_index from a nightly version of pytorch.
# https://github.com/pytorch/pytorch/blob/5292a92e03f6f33ba8363abcae708c943baaf275/torch/functional.py#L1692
# 
# This is a temporary workaround for the fact that torch.unravel_index is not currently in the stable version of pytorch.
# CHECK BACK in a few weeks/months to see if this is still necessary and if any bugs were found/changes were made
# that should be mirrored here.
#############################

from typing import (
    List, Tuple, Optional, Union, Any, Sequence, TYPE_CHECKING
)
import operator
import itertools

import torch

from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)

Tensor = torch.Tensor

def unravel_index(indices: Tensor, shape: Union[int, Sequence[int], torch.Size]) -> List[Tensor]:
    r"""Converts a tensor of flat indices into a tuple of coordinate tensors that
    index into an arbitrary tensor of the specified shape.

    Args:
        indices (Tensor): An integer tensor containing indices into the
            flattened version of an arbitrary tensor of shape :attr:`shape`.
            All elements must be in the range ``[0, prod(shape) - 1]``.

        shape (int, sequence of ints, or torch.Size): The shape of the arbitrary
            tensor. All elements must be non-negative.

    Returns:
        tuple of Tensors: Each ``i``-th tensor in the ouput corresponds with
        dimension ``i`` of :attr:`shape`. Each tensor has the same shape as
        ``indices`` and contains one index into dimension ``i`` for each of the
        flat indices given by ``indices``.

    Example::

        >>> import torch
        >>> torch.unravel_index(torch.tensor(4), (3, 2))
        (tensor(2),
         tensor(0))

        >>> torch.unravel_index(torch.tensor([4, 1]), (3, 2))
        (tensor([2, 0]),
         tensor([0, 1]))

        >>> torch.unravel_index(torch.tensor([0, 1, 2, 3, 4, 5]), (3, 2))
        (tensor([0, 0, 1, 1, 2, 2]),
         tensor([0, 1, 0, 1, 0, 1]))

        >>> torch.unravel_index(torch.tensor([1234, 5678]), (10, 10, 10, 10))
        (tensor([1, 5]),
         tensor([2, 6]),
         tensor([3, 7]),
         tensor([4, 8]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (10, 10, 10, 10))
        (tensor([[1], [5]]),
         tensor([[2], [6]]),
         tensor([[3], [7]]),
         tensor([[4], [8]]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (100, 100))
        (tensor([[12], [56]]),
         tensor([[34], [78]]))
    """
    if has_torch_function_unary(indices):
        return handle_torch_function(
            unravel_index, (indices,), indices, shape=shape)
    res_tensor = _unravel_index(indices, shape)
    return res_tensor.unbind(-1)

def _unravel_index(indices: Tensor, shape: Union[int, Sequence[int]]) -> Tensor:
    assert(not indices.is_complex() and not indices.is_floating_point() and not indices.dtype == torch.bool)
    # "expected 'indices' to be integer dtype, but got {indices.dtype}"

    assert(isinstance(shape, (int, Sequence)))
    # "expected 'shape' to be int or sequence of ints, but got {type(shape)}"

    if isinstance(shape, int):
        shape = torch.Size([shape])
    else:
        for dim in shape:
            assert(isinstance(dim, int))
            # "expected 'shape' sequence to only contain ints, but got {type(dim)}"
        shape = torch.Size(shape)

    assert(all(dim >= 0 for dim in shape))
    # "'shape' cannot have negative values, but got {tuple(shape)}"

    coefs = list(reversed(list(itertools.accumulate(reversed(shape[1:] + torch.Size([1])), func=operator.mul))))
    res = indices.unsqueeze(-1).div(
        torch.tensor(coefs, device=indices.device, dtype=torch.int64),
        rounding_mode='trunc') % torch.tensor(shape, device=indices.device, dtype=torch.int64)

    return res