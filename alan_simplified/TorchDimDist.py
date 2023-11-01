import torch as t
import torch.distributions as td
import functorch.dim
from functorch.dim import Dim

from .utils import *
Tensor = (t.Tensor, functorch.dim.Tensor)

def generic_tdd_order(a, dims: list[Dim], event_ndim: int):
    if isinstance(a, functorch.dim.Tensor):
        return tdd_order(a, dims, event_ndim)
    else:
        return a

def tdd_order(a: Tensor, dims: list[Dim], event_ndim: int):
    """
    Rearranges tensor as:
    a[..., dims, :, :]
    where there are event_ndim colons at the end, and 
    """

    #Puts the torchdims at the front, and adds singleton dimensions where there is a
    #dim in dims but not in a.dims.
    a = singleton_order(a, dims)
    a_ndim = a.ndim

    torchdim_dims     = [*range(len(dims))]
    batch_unnameddims = [*range(len(dims), a_ndim-event_ndim)]
    event_unnameddims = [*range(a_ndim-event_ndim, a_ndim)]

    return a.permute(*batch_unnameddims, *torchdim_dims, *event_unnameddims)

def colons(n: int):
    return n*[slice(None)]

class TorchDimDist():
    r"""
    Wrapper for PyTorch dists to make them accept TorchDim arguments.

    :class:`TorchDimDist` allows for sampling (or evaluating the log probability of) TorchDim-ed tensors
    from distributions with non-dimmed arguments as well as sampling from distributions with dimmed arguments
    
    Note that at present there is no sample_shape dimension, to do IID sampling over
    new non-torchdim dimensions.  To achieve the same effect, do something like
    ```
    alan.Normal(t.zeros(3)[:, None].expand(-1, 4), 1)
    ```

    .. warning::
    For people editting the class in future: self.dist and self.dims are exposed!
    """
    def __init__(self, dist, **kwargs):
        r"""
        Creates a TorchDimDist.

        Args:
            dist : PyTorch distribution
            kwargs (Dict): torchdim tensor arguments to the distribution.  Must be named.
        """
        self.dist = dist
        self.kwargs_torchdim = kwargs

        #Extract the dimension of the events and arguments from the underlying PyTorch distribution.
        #For instance, in MultivariateNormal, we get vectors as samples (event_dim=1),
        #The mean is vector (event_dim=1) and covariance is a matrix (event_dim=2).
        self.sample_event_dim = dist.support.event_dim
        self.arg_event_dim = {arg: dist.arg_constraints[arg].event_dim for arg in kwargs}



    def sample(self, reparam: bool, sample_dims: list[Dim], sample_shape):
        r"""
        Samples, making sure the resulting sample has all the dims in sample_dims, 
        and has the unnamed shape from self.sample_shape.

        Args:
            reparam (bool): *True* for reparameterised sampling (Not supported by all dists)
            sample_dims: _all_ dimensions in the resulting samples (not just the extra dims)
                         should include all the dims in the input.
            sample_shape: unnamed/integer extra samples.

        Returns:
            sample (torchdim Tensor): sample with correct dimensions
        """
        #Check that all the dimensions in sample_dims are unique.
        assert_unique_dim_iter(sample_dims, 'sample_dims')

        #List of torchdims in arguments
        arg_dims = unify_dims(self.kwargs_torchdim.values())

        #Check that all the torchdims on the arguments are in sample_dims.
        set_sample_dims = set(sample_dims)
        for dim in arg_dims:
            assert dim in set_sample_dims

        #Dims in sample_dims that aren't in arg_dims
        extra_sample_dims = list(set(sample_dims).difference(arg_dims))
        extra_sample_dim_sizes = [esd.size for esd in extra_sample_dims]
        
        kwargs_tensor = {}
        for name, arg_torchdim in self.kwargs_torchdim.items():
            #Rearrange tensors as 
            #[unnamed batch dimensions, torchdim batch dimensions, event dimensions]
            kwargs_tensor[name] = generic_tdd_order(arg_torchdim, arg_dims, self.arg_event_dim[name])

        if reparam and not self.dist.has_rsample:
            raise Exception(f'Trying to do reparameterised sampling of {type(self.dist)}, which is not implemented by PyTorch (likely because {type(self.dist)} is a distribution over discrete random variables).')

        dist = self.dist(**kwargs_tensor)
        sample_method = getattr(dist, "rsample" if reparam else "sample")

        #sample_shape = [named batch dims, unnamed batch dims]
        sample_shape = [*sample_shape, *extra_sample_dim_sizes]
        sample_tensor = sample_method(sample_shape=sample_shape)
        
        #output dims are:
        #[unnamed_batch_dims, extra_torchdims, arg_torchdims, unnamed_event_dims]
        dims = [..., *extra_sample_dims, *arg_dims, *colons(self.sample_event_dim)]
        return generic_getitem(sample_tensor, dims)



    def log_prob(self, x):
        assert isinstance(x, Tensor)

        dims = unify_dims([x, *self.kwargs_torchdim.values()])

        x_tensor = tdd_order(x, dims, self.sample_event_dim)

        kwargs_tensor = {}
        for name, arg_torchdim in self.kwargs_torchdim.items():
            #Rearrange tensors as 
            #[unnamed batch dimensions, torchdim batch dimensions, event dimensions]
            kwargs_tensor[name] = generic_tdd_order(arg_torchdim, dims, self.arg_event_dim[name])
        dist = self.dist(**kwargs_tensor)
        lp_tensor = dist.log_prob(x_tensor)

        return generic_getitem(lp_tensor, [..., *dims])

