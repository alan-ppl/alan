import torch

class BufferStore(torch.nn.Module):
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
            assert isinstance(v, torch.Tensor)
            self.keys.append(k)

            self.register_buffer(k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys}

class ParameterStore(torch.nn.Module):
    """
    Holds buffers in a dictionary
    makes sure they e.g. get moved to device correctly.
    """

    def __init__(self, paramdict:dict):
        super().__init__()
        restricted_keys = dir(self)

        self.keys = []
        for k, v in paramdict.items():
            assert k not in restricted_keys
            assert isinstance(v, torch.Tensor)
            self.keys.append(k)

            if not isinstance(v, torch.nn.Parameter):
                v = torch.nn.Parameter(v)
            self.register_parameter(k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys}

class ModuleStore(torch.nn.Module):
    def __init__(self, moddict:dict):
        super().__init__()
        restricted_keys = dir(self)

        self.keys = []
        for k, v in moddict.items():
            assert k not in restricted_keys
            assert isinstance(v, torch.nn.Module)
            self.keys.append(k)
            self.register_module(k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys}
