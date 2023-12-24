import torch as t
import torch.distributions as td
import functorch.dim
from functorch.dim import Dim

from .utils import *
Tensor = (t.Tensor, functorch.dim.Tensor)

def generic_order_arg(a, torchdims: list[Dim], batch_ndim:int, event_ndim:int):
    if isinstance(a, functorch.dim.Tensor):
        return order_arg(a, torchdims, batch_ndim, event_ndim)
    else:
        return a

def order_arg(a: Tensor, torchdims: list[Dim], batch_ndim:int, event_ndim:int):
    """
    Inputs a torchdim Tensor (either a distribution parameter or sample), with positional shape:
    [*batch_shape, *event_shape]

    Here:
    * batch_ndim is the number of unnamed, non-event dimensions (arising from unnamed dimensions in params)
    * event_ndim is the number of dimensions in the parameters for a single sample (e.g. 1 for MvN mean, 2 for MvN cov).

    Rearranges tensor as:
    a[*batch_shape, *torchdims, *event_shape]
    """
    torch_ndim = len(torchdims)

    #Check that all dimensions in a are present in torchdims:
    assert set(generic_dims(a)).issubset(torchdims)

    #Check that length of tensor is correct:
    assert generic_ndim(a) == batch_ndim + event_ndim

    #[*torchdim_shape, *batch_shape, *event_shape]
    a = singleton_order(a, torchdims)

    assert a.ndim == torch_ndim + batch_ndim + event_ndim
    torchdim_idxs     = [*range(torch_ndim)]
    batch_idxs = [*range(torch_ndim, torch_ndim + batch_ndim)]
    event_idxs    = [*range(torch_ndim + batch_ndim, torch_ndim + batch_ndim + event_ndim)]

    return a.permute(*batch_idxs, *torchdim_idxs, *event_idxs)

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
        #List of torchdims in arguments
        self.all_arg_dims = unify_dims(self.kwargs_torchdim.values())
        self.set_all_arg_dims = set(self.all_arg_dims)

        #Extract the dimension of the events and arguments from the underlying PyTorch distribution.
        #For instance, in MultivariateNormal, we get vectors as samples (event_dim=1),
        #The mean is vector (event_dim=1) and covariance is a matrix (event_dim=2).
        self.sample_event_ndim = dist.support.event_dim
        self.arg_event_ndim = {arg: dist.arg_constraints[arg].event_dim for arg in kwargs}

        #Dict of unnamed batch dimensions, extracted from the size of the unnamed arguments.
        self.arg_batch_ndim = {arg: generic_ndim(self.kwargs_torchdim[arg]) - self.arg_event_ndim[arg] for arg in kwargs}
        self.sample_batch_ndim = max(self.arg_batch_ndim.values())

        # Distribution constructed with arguments of shape:
        # [batch_shape, all_arg_dims, event_shape]
        # not that all_arg_dims is shared for all args, while the batch shape could be different.

        self.kwargs_tensor = {}
        for name, arg_torchdim in self.kwargs_torchdim.items():
            self.kwargs_tensor[name] = generic_order_arg(
                arg_torchdim, 
                self.all_arg_dims,
                batch_ndim=self.arg_batch_ndim[name],
                event_ndim=self.arg_event_ndim[name]
            )

        self.dist_tensor = self.dist(**self.kwargs_tensor)

        self.batch_arg_event_dims = [
            *colons(self.sample_batch_ndim),  # batch_shape
            *self.all_arg_dims,               # all_arg_dims
            *colons(self.sample_event_ndim),   # event_shape
        ]

        self.batch_arg_dims = [
            *colons(self.sample_batch_ndim),  # batch_shape
            *self.all_arg_dims,               # all_arg_dims
        ]

    def extra_dims(self, sample_dims:list[Dim]):
        #Check that all the dimensions in sample_dims are unique.
        assert_unique_dim_iter(sample_dims, 'sample_dims')

        #Split sample_dims into extra dimensions and dimensions in sample_dims.
        extra_dims = list(set(sample_dims).difference(self.set_all_arg_dims))

        return extra_dims

    def sample(self, reparam: bool, sample_dims: list[Dim], sample_shape):
        r"""
        Samples, making sure the resulting sample has all the dims in sample_dims, 
        and has the unnamed shape from self.sample_shape.

        Args:
            reparam (bool): *True* for reparameterised sampling (Not supported by all dists)
            sample_dims: _all_ TorchDim dimensions in the resulting samples (not just the extra dims)
                         should include all the dims in the input.
            sample_shape: unnamed/integer extra samples.

        Returns:
            sample (torchdim Tensor): sample with correct dimensions
        """
        #Check that all the torchdims on the arguments are in sample_dims.
        assert set(self.set_all_arg_dims).issubset(sample_dims)

        if reparam and not self.dist.has_rsample:
            raise Exception(f'Trying to do reparameterised sampling of {type(self.dist)}, which is not implemented by PyTorch (likely because {type(self.dist)} is a distribution over discrete random variables).')

        extra_dims = self.extra_dims(sample_dims)
        assert set(sample_dims) == self.set_all_arg_dims.union(extra_dims)
        extra_shape = [esd.size for esd in extra_dims]

        sample_method = getattr(self.dist_tensor, "rsample" if reparam else "sample")

        #[*sample_shape, *extra_shape, *batch_shape, *all_arg_shape, *event_shape]
        sample_tensor = sample_method(sample_shape=[*sample_shape, *extra_shape])

        dims = [
            *colons(len(sample_shape)),        # sample_shape
            *extra_dims,                       # extra_dims
            *self.batch_arg_event_dims, # everythin else
        ]
        return generic_getitem(sample_tensor, dims)



#    def log_prob(self, x):
#        """
#        This is subtle, because args can have lots of K-dimensions that aren't on x.
#        Therefore, we use singleton_order, which gives singleton dimensions in x_tensor
#        for dimensions that are in args, but not x.
#        """
#        assert isinstance(x, Tensor)
#
#        sample_dims = generic_dims(x)
#        extra_dims = self.extra_dims(sample_dims)
#
#        batch_ndim = self.sample_batch_ndim
#        event_ndim = self.sample_event_ndim
#        sample_ndim = generic_ndim(x) - batch_ndim - event_ndim
#
#        dims = [
#            *colons(sample_ndim),
#            *extra_dims,
#            *self.lp_batch_arg_dims,
#        ]
#
#        x_tensor = singleton_order(x, dims)
#        lp_tensor = self.dist_tensor.log_prob(x_tensor)
#        
#        lp = generic_getitem(lp_tensor, dims)
#
#        return sum_non_dim(lp)

    def log_prob(self, x):
        """
        This is subtle, because args can have lots of K-dimensions that aren't on x.
        Therefore, we use singleton_order, which gives singleton dimensions in x_tensor
        for dimensions that are in args, but not x.

        Remember that x comes in as a torchdim tensor with positional dims:
        [*sample_shape, *batch_shape, *event_shape]
        """
        assert isinstance(x, Tensor)

        sample_dims = generic_dims(x)  #Extra dims + a subset of all_arg_dims.
        extra_dims = self.extra_dims(sample_dims)

        batch_ndim = self.sample_batch_ndim
        event_ndim = self.sample_event_ndim
        sample_ndim = generic_ndim(x) - batch_ndim - event_ndim

        x_dims = [
            *colons(sample_ndim),
            *extra_dims,
            *self.batch_arg_event_dims,
        ]

        lp_dims = [
            *colons(sample_ndim),
            *extra_dims,
            *self.batch_arg_dims,
        ]

        x_tensor = singleton_order(x, x_dims)
        lp_tensor = self.dist_tensor.log_prob(x_tensor)
        
        lp = generic_getitem(lp_tensor, lp_dims)

        return sum_non_dim(lp)

