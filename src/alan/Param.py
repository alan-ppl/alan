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
    def __init__(self, init, transformation=None, ignore_platenames=(), name=None):
        if transformation is None:
            transformation = identity

        self.init = proc_init(init)
        self.trans = transformation
        self.ignore_platenames = ignore_platenames
        self.name = name

class QEMParam(Param):
    def __init__(self, init, ignore_platenames=(), name=None):
        self.init = proc_init(init)
        self.trans = identity
        self.ignore_platenames = ignore_platenames
        self.name = name

