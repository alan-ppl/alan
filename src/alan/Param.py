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
    def __init__(self, init, trans=identity, ignore_platenames=()):
        self.init = proc_init(init)
        self.trans = trans
        self.ignore_platenames = ignore_platenames

class QEMParam(Param):
    def __init__(self, init, ignore_platenames=()):
        self.init = proc_init(init)
        self.trans = identity
        self.ignore_platenames = ignore_platenames

