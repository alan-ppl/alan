from .utils import *

class GroupSample():
    def __init__(self, sample:dict[str, Tensor]):
        self.sample = sample
