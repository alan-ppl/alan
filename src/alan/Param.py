class Param:
    def __init__(self, init, trans=None, platenames=None):
        self.init = init
        self.trans = trans
        self.platenames = platenames

class OptParam(Param):
    pass

class QEMParam(Param):
    pass

