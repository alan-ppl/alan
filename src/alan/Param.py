class Param:
    pass

def identity(x):
    return x

class OptParam(Param):
    def __init__(self, init, trans=identity, ignore_platenames=()):
        self.init = init
        self.trans = trans
        self.ignore_platenames = ignore_platenames

class QEMParam(Param):
    def __init__(self, init, ignore_platenames=()):
        self.init = init
        self.trans = identity
        self.ignore_platenames = ignore_platenames

