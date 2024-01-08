import torch

class TensorStore(torch.nn.Module):
    def __init__(self, bufferdict:dict):
        super().__init__()
        self.keys = []
        self.names = {}

        restricted_keys = dir(self)

        for k, v in bufferdict.items():
            assert k not in restricted_keys
            self.keys.append(k)

            assert isinstance(v, torch.Tensor)
            self.register_tensor(k, v)

            self.names[k] = v.names

    def to_dict(self):
        return {k: getattr(self, k).refine_names(*self.names[k]) for k in self.keys}

class BufferStore(TensorStore):
    """
    Holds buffers in a dictionary
    makes sure they e.g. get moved to device correctly.
    """
    def register_tensor(self, k, v):
        self.register_buffer(k, v)


class ParameterStore(TensorStore):
    """
    Holds buffers in a dictionary
    makes sure they e.g. get moved to device correctly.
    """
    def register_tensor(self, k, v):
        if not isinstance(v, torch.nn.Parameter):
            v = torch.nn.Parameter(v)
        self.register_parameter(k, v)


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
