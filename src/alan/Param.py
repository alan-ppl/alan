import torch as t

from .utils import Number

class Param:
    pass

def identity(x):
    return x

def proc_init(init):
    if isinstance(init, Number):
        init = t.tensor(float(init))
    assert isinstance(init, t.Tensor)
    return init

class OptParam(Param):
    """Describes the initial value of a parameter to be optimized (e.g. using VI).

    Note that this class doesn't know the eventual size of the parameter, as it doesn't know the platesizes.  Instead, the actual parameter gets initialized when the Plate is wrapped in a BoundPlate, as that's when the platesizes become known.

    Example use:

    .. code-block:: python

       Plate(
           ...
           a = Normal(OptParam(0.), OptParam(0., transformation=t.exp),
           ...
       )

    Arguments:
        init (float or Tensor): 
            Initial value of the parameter.  Usually, this would be a float.  If you wanted to specify e.g. the mean of a MultivariateNormal, you'd use a torch.Tensor.  But this would always be a plain tensor, never named.

    Keyword Arguments:
        trans (function): 
            Transformation to be applied to the parameter before use.  This is most useful for e.g. the scale of a Gaussian, which must be postive.  But you can't guarantee that the optimizer will keep the parameter positive.  So instead, you apply a transformation, such as exponentiation, which maps any number on the real line to something positive.  Note that the ``init`` is the initial value _before_ the transformation.
        ignore_platenames (iterable): 
            By default, we create a parameter with all appropriate platenames.  Such parameters could be quite large, and this could be inappropriate if we want to set up parameter on the prior, P.  So if you want to skip any parameters, you'd include them in here.
        name (str):
            By default, the parameter is named <variable name>_<distribution argument name>.  So in the example above, we'd end up with two parameters, named ``a_loc`` and ``a_scale``.
    """
    def __init__(self, init, transformation=None, ignore_platenames=(), name=None):
        if transformation is None:
            transformation = identity

        self.init = proc_init(init)
        self.trans = transformation
        self.ignore_platenames = ignore_platenames
        self.name = name

class QEMParam(Param):
    """Describes the initial value of a parameter to be learned using QEM.

    Note that this class doesn't know the eventual size of the parameter, as it doesn't know the platesizes.  Instead, the actual parameter gets initialized when the Plate is wrapped in a BoundPlate, as that's when the platesizes become known.

    Example use:

    .. code-block:: python

       Plate(
           ...
           a = Normal(QEMParam(0.), QEMParam(1.))
           ...
       )

    Note that unlike ``OptParam``, ``QEMParam`` doesn't need transformations to ensure that learned parameters remain sensible.

    Arguments:
        init (float or Tensor): 
            Initial value of the parameter.  Usually, this would be a float.  If you wanted to specify e.g. the mean of a MultivariateNormal, you'd use a torch.Tensor.  But this would always be a plain tensor, never named.

    Keyword Arguments:
        ignore_platenames (iterable): 
            By default, we create a parameter with all appropriate platenames.  Such parameters could be quite large, and this could be inappropriate if we want to set up parameter on the prior, P.  So if you want to skip any parameters, you'd include them in here.
        name (str):
            By default, the parameter is named <variable name>_<distribution argument name>. So in the example above, we'd end up with two parameters, named ``a_loc`` and ``a_scale``.
    """
    def __init__(self, init, ignore_platenames=(), name=None):
        self.init = proc_init(init)
        self.trans = identity
        self.ignore_platenames = ignore_platenames
        self.name = name

