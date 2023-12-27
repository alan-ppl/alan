import torch

class BufferDict(torch.nn.Module):
    """
    Holds buffers in a dictionary
    makes sure they e.g. get moved to device correctly.
    """

    def __init__(self, bufferdict:dict):
        super().__init__()
        restricted_keys = dir(self)

        self.keys = []
        for k, v in bufferdict.items():
            assert k not in restricted_keys
            self.keys.append(k)
            self.register_buffer(k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys}
