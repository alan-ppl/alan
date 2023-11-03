from typing import Optional
from dist import Dist

class Group():
    def __init__(self, **kwargs):
        #Groups can only contain Dist, not Plates/Timeseries/other Groups.
        for dist in kwargs.values():
            assert isinstance(dist, Dist)

        self.prog = kwargs
        self.all_args = list(set([dist.all_args for dist in self.kwargs.values()]))

    def resampled_scope(
            self, 
            scope:dict[str, Tensor], 
            active_platedims:list[Dim], 
            Kdim:Dim, 
            sampling_type:SamplingType):

        resampled_scope = {}
        for (name, tensor) in scope.items():
            if name in self.all_args:
                resampled_scope[name] = sampling_type.resample(tensor, Kdim, active_platedims)
        return resampled_scope

    def sample(self, 
               scope: dict[str, Tensor], 
               active_platedims: list[Dim], 
               Kdim:Optional[Dim],
               sampling_type:SamplingType,
               reparam=True):

        sample_dims = [Kdim, *active_platedims]

        scope = self.resampled_scope(
            scope=scope,
            active_platedims=active_platedims,
            Kdim=Kdim,
            sampling_type=sampling_type
        )

        result = {}
        for name, dist in self.prog.items():
            tdd = dist.tdd(scope)
            sample = tdd.sample(reparam, sample_dims, dist.sample_shape)

            scope[name] = scope
            result[name] = result

        return result

    def log_prob(self, 
                 sample: Tensor, 
                 scope: dict[any, Tensor], 
                 active_platedims: list[str], 
                 Kdim: Optional[Dim],
                 sampling_type:SamplingType):

        total_log_prob = 0.
        for name, dist in self.prog.items():
            tdd = dist.tdd(scope)
            lp = tdd.log_prob(sample[name])
            total_log_prob = total_log_prob + lp

        return total_log_prob
