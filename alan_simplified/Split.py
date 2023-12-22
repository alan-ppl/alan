from .utils import *
import math

class NoSplit:
    def split_args(self, name, sample, inputs_params, extra_log_factors, data, all_platedims):
        return [{
            'sample':sample, 
            'inputs_params':inputs_params, 
            'extra_log_factors':extra_log_factors, 
            'data':data,
            'all_platedims':all_platedims,
        }]

class NoCheckpoint(NoSplit):
    pass
no_checkpoint = NoCheckpoint()

class Checkpoint(NoSplit):
    pass
checkpoint = Checkpoint()

class Split:
    def __init__(self, platename:str, split_size:int):
        self.platename = platename
        assert isinstance(split_size, int)
        self.split_size = split_size

    def splitdims(self, all_platedims):
        return SplitDims(self, all_platedims)

    def split_args(self, name, sample, inputs_params, extra_log_factors, data, all_platedims):
        if self.platename == name:
            split = self.splitdims(all_platedims)

            samples            = split.split_dict(sample)
            inputs_paramss     = split.split_dict(inputs_params)
            extra_log_factorss = split.split_dict(extra_log_factors)
            datas              = split.split_dict(data)
            all_platedimss     = split.split_all_platedimss
        else:
            samples            = [sample]
            inputs_paramss     = [inputs_params]
            extra_log_factorss = [extra_log_factors]
            datas              = [data]
            all_platedimss     = [all_platedims]

        del sample, inputs_params, extra_log_factors, data, all_platedims

        result = []
        for (s, i, e, d, a) in zip(samples, inputs_paramss, extra_log_factorss, datas, all_platedimss):
            result.append({
                'sample' : s,
                'inputs_params' : i,
                'extra_log_factors' : e,
                'data' : d,
                'all_platedims' : a,
            })
        return result




class SplitDims:
    def __init__(self, split: Split, all_platedims:dict[str, Dim]):
        self.split = split

        self.orig_dim = all_platedims[split.platename]
        orig_size = self.orig_dim.size
        split_size = self.split.split_size

        assert orig_size > split_size

        self.split_sizes = [*((orig_size//split_size)*[split_size]), orig_size%split_size]
        self.split_dims = [Dim(f'{self.split.platename}_split_{i}', self.split_sizes[i]) for i in range(len(self.split_sizes))]
        self.split_all_platedimss = [{**all_platedims, self.split.platename: dim} for dim in self.split_dims]


    def split_tensor(self, x:Tensor):
        non_split_dims = [dim for dim in generic_dims(x) if dim is not self.orig_dim]

        xs = generic_order(x, [self.orig_dim, *non_split_dims]).split(self.split_sizes)
        return [generic_getitem(x, [split_dim, *non_split_dims]) for (x, split_dim) in zip(xs, self.split_dims)]

    def split_dict(self, d:dict):
        """
        User-facing method, that takes a tree-structured dict, and returns a split tuple of tree-structured dicts.
        """
        results = [{} for _ in self.split_sizes]
        self._split_dict_inner(d, results)
        return results

    def _split_dict_inner(self, d:dict, results:list[dict]):
        for (k, v) in d.items():
            if isinstance(v, dict):
                new_results = []
                for result in results:
                    result[k] = {}
                    new_results.append(result[k])

                self._split_dict_inner(v, new_results)
            else:
                assert isinstance(v, Tensor)
                for result, val in zip(results, self.split_tensor(v)):
                    result[k] = val

nosplit = Split(math.nan, 0)
