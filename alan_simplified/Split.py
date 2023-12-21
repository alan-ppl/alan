from .utils import *

class Split:
    def __init__(self, platename:str, split_size:int):
        self.platename = platename
        self.split_size = split_size

    def split_tensor(self, x:Tensor, all_platedims:dict[str, Dim]):
        dim = all_platedims[self.platename]
        return x.order(dim).split(self.split_size)



