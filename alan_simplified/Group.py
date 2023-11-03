from typing import Optional
from dist import Dist, filter_resample_scope

class Group():
    def __init__(self, **kwargs):
        #Groups can only contain Dist, not Plates/Timeseries/other Groups.
        for dist in kwargs.values():
            assert isinstance(dist, Dist)

        self.prog = kwargs
        self.all_args = list(set([dist.all_args for dist in self.kwargs.values()]))

    def sample(
            self,
            name:Optional[str],
            scope: dict[str, Tensor], 
            active_platedims:list[Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampling_type:SamplingType,
            reparam:bool):

        Kdim = groupvarname2Kdim[name]
        sample_dims = [Kdim, *active_platedims]

        resampled_scope = filter_resample_scope(
            all_args=self.all_args
            scope=scope,
            active_platedims=active_platedims,
            Kdim=Kdim,
            sampling_type=sampling_type
        )

        result = {}
        for name, dist in self.prog.items():
            tdd = dist.tdd(resampled_scope)
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
