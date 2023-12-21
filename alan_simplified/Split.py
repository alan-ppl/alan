from .utils import *

class Split:
    def __init__(self, platename:str, split_size:int):
        self.platename = platename
        self.split_size = split_size

    def split_tensor(self, x:Tensor, all_platedims:dict[str, Dim]):
        """
        Returns a tensor split into pieces.
        Splitting the tensor also implies that we need a different platedimension,
        so also return the corresponding all_platedims.
        """
        dim = all_platedims[self.platename]
        xs_tensor = x.order(dim).split(self.split_size)

        dims = []
        xs_dim = []
        modified_all_platedimss = []

        for x in xs_tensor:
            dim = Dim(self.platename, x.shape[0])
            dims.append(dim)
            xs_dim.append(x[dim])
            modified_all_platedims = {**all_platedims}
            modified_all_platedims[self.platename] = dim
            modified_all_platedimss.append(modified_all_platedims)

        return xs_dim, modified_all_platedimss
