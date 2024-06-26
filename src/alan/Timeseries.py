import torch.nn as nn
from .dist import _Dist, sample_gdt
from .utils import *
from .Sampler import Sampler

from typing import Optional



#Checks to Timeseries init.
#if 0 == len(active_platedims):
#    raise Exception(f"Timeseries can't be in the top-layer plate, as there's no platesize at the top")
#if name not in self.trans.all_args:
#    raise Exception(f"The timeseries transition distribution for {name} must have some dependence on the previous timestep; you get that by including {name} as an argument in the transition distribution.")


class Timeseries(nn.Module):
    """
    In progress!!!!

    See `examples/timeseries.py`

    Arguments:
        init (str):
            string, representing the initial state as a random variable.  This random variable must have been sampled in the immediately above plate.

        trans (Dist):
            transition distribution.

    As an example:

    .. code-block:: python

       Plate(
           ts_init = Normal(0., 1.),
           T = Plate(
               ts = Timeseries('ts_init', Normal(lambda ts: 0.9*ts, 0.1)),
           )
       )

    In the exmplae:

    * ``T`` is the plate (i.e. ``all_platesizes['T']``) is the length of the timeseries.  Note that this is a slight abuse of the term "Plate", which is usually only used to refer to independent variables.
    * ``ts`` is the name of the timeseries random variable itself.
    * ``Normal(lambda ts: 0.9*ts, 0.1)`` is the transition distribution.  Note that it refers to the previous step of itself using the timeseries name itself, ``ts``, as an argument.
    * ``ts_init`` is the initial state. Must be a string representing a random variable in the previous plate.

    Non-split implementation notes:

    * Non-split log_PQ_plate returns a K_ts_init tensor.
    * Splitting log_PQ_plate:

      - Uses a backward pass, so at the start of the backward pass, we sum from the back.
      - At the start of the backward pass, log_PQ_plate takes one unusual input: initial timeseries state, with dimension K_ts.  If initial timeseries state is provided as a kwarg, we ignore Timeseries.init.
      - log_PQ_plate returns K_ts dimensional tensor, resulting from summing all the way from the back to the start of the split.
      - The next split takes two unusual arguments: the initial state, and the log_pq from the last split evaluated by the backward pass.

    Note:   
       You can't currently split along a timeseries dimension (and you may never be able to).

    Note:
       OptParam and QEMParam are currently banned in timeseries.
    """
    def __init__(self, init, trans):
        super().__init__()

        self.qem_dist = False
        self.is_timeseries = True

        if not isinstance(init, str):
            raise Exception(f"the first / `init` argument in a Timeseries should be a string, representing a variable name in the above plate")

        if not isinstance(trans, _Dist):
            raise Exception("the second / `trans` argument in a Timeseries should be a distribution")

        if trans.sample_shape != t.Size([]):
            raise Exception("sample_shape on the transition distribution must not be set; if you want a sample_shape, it needs to be on the initial state")

        self.init = init
        self.trans = trans.finalize(None)
        assert not self.trans.qem_dist
        #Will include own name, but that'll be eliminated in the first step of sample_gdt
        self.all_args = [init, *self.trans.all_args] 

    @property
    def opt_qem_params(self):
        return self.trans.opt_qem_params

    def sample(self, scope, reparam: bool, active_platedims:list[Dim], K_dim:Dim, timeseries_perm):
        assert 0 <= len(active_platedims)
        (other_platedims, T_dim) = (active_platedims[:-1], active_platedims[-1])

        #Set previous state equal to initial state.
        prev_state = scope[self.init]
        #Check that prev_state has the right dimensions
        if set(prev_state.dims) != set([K_dim, *other_platedims]):
            raise Exception(f"Initial state, {self.init}, doesn't have the right dimensions for timeseries {name}; the initial state must be defined one step up in the plate heirarchy")

        sample_timesteps = []

        for time in range(T_dim.size):
            #new scope, where we select out the time'th timestep for any tensor with a time dimension.
            #all these variables have already been resampled.
            timeseries_scope = {}
            for k, v in scope.items():
                if T_dim in set(generic_dims(v)):
                    v = v.order(T_dim)[time]
                timeseries_scope[k] = v
            #Put previous timestep for self into scope
            timeseries_scope['prev'] = prev_state


            #sample the next timestep
            sample_timestep = self.trans.sample(timeseries_scope, reparam, other_platedims, K_dim, None)
            sample_timesteps.append(sample_timestep)

            #Permute this timestep, ready for being used as prev_state.
            if timeseries_perm is not None:
                timestep_perm = timeseries_perm.order(T_dim)[time]
                sample_timestep = sample_timestep.order(K_dim)[timestep_perm, ...][K_dim]
            prev_state = sample_timestep

        return t.stack(sample_timesteps, 0)[T_dim]

    def sample_extended(self,
            sample:Tensor,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:dict):
        
        (active_extended_platedims, extended_T_dim) = (active_extended_platedims[:-1], active_extended_platedims[-1])
        
        original_T_dim = original_platedims[str(extended_T_dim)]
        #Number of timesteps to roll forward:
        timesteps = extended_T_dim.size - original_T_dim.size

        original_platedims = [original_platedims[str(dim)] for dim in active_extended_platedims]
        #set prev state to last timestep of sample
        prev_state = sample.order(original_T_dim)[-1].order(*original_platedims)[active_extended_platedims]
        
        #Don't need to check prev_state has right dims...
        
        extended_sample_timesteps = []
        

        for time in range(timesteps):
            
            timeseries_scope = {}
            for k, v in scope.items():
                if extended_T_dim in set(generic_dims(v)):
                    v = v.order(extended_T_dim)[time]
                timeseries_scope[k] = v
                
            timeseries_scope['prev'] = prev_state
            
            #sample the next timestep
            sample_timestep = self.trans.sample(timeseries_scope, reparam, active_extended_platedims, Ndim, None)
            extended_sample_timesteps.append(sample_timestep)
            
            prev_state = sample_timestep
            
        #stack old and new timesteps
        # print(sample.order(original_T_dim))
        # print(t.stack(extended_sample_timesteps, 0))
        
        original_sample = sample.order(original_T_dim)
        original_sample_new_platedims = original_sample.order(*original_platedims)[active_extended_platedims]
        
        return t.cat([original_sample_new_platedims, t.stack(extended_sample_timesteps, 0)], 0)[extended_T_dim]

    def predictive_ll(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            original_data:dict[str, Tensor],
            extended_data:dict[str, Tensor]):
        
        original_ll, extended_ll = {}, {}

        if name in extended_data.keys():
            extended_ll[name], _ = self.log_prob(extended_data[name], scope, None, None)

            original_dims, extended_dims = corresponding_plates(original_platedims, extended_platedims, original_data[name], extended_data[name]) 

            # Take the logprob of the original data from the extended logprob tensor
            original_idxs = [slice(0, dim.size) for dim in original_dims]
            original_ll[name] = generic_getitem(generic_order(extended_ll[name], extended_dims), original_idxs)
            original_ll[name] = generic_getitem(original_ll[name], original_dims)

        return original_ll, extended_ll
    
        
    def log_prob(self, sample, scope:dict, T_dim:Dim, K_dim:Dim):

        assert isinstance(scope, dict)
        assert isinstance(sample, Tensor)
        assert isinstance(T_dim, Dim)
        assert isinstance(K_dim, Dim)

        set_dims_sample = set(generic_dims(sample))
        assert K_dim in set_dims_sample
        assert T_dim in set_dims_sample

        initial_state = scope[self.init]
        set_dims_initial = set(generic_dims(initial_state))
        assert T_dim not in set_dims_initial
        assert len(set_dims_initial) + 1 == len(set_dims_sample)

        diff_dims = list(set_dims_initial.difference(set_dims_sample))
        assert 1 == len(diff_dims)
        Kinit_dim = diff_dims[0]
        assert Kinit_dim in set_dims_initial

        sample_prev = sample.order(K_dim)[Kinit_dim]

        sample_prev = t.cat([
            initial_state[None, ...],
            sample_prev.order(T_dim)[:-1],
        ], 0)[T_dim]
        set_dims_prev_sample = set(generic_dims(sample_prev))
        assert Kinit_dim in set_dims_prev_sample
        assert K_dim not in set_dims_prev_sample
        assert T_dim     in set_dims_prev_sample

        scope = {**scope}
        scope['prev'] = sample_prev

        lp, _ = self.trans.log_prob(sample, scope, None, None)
        set_dims_lp = set(generic_dims(lp))

        assert Kinit_dim in set_dims_lp
        assert K_dim in set_dims_lp
        assert T_dim in set_dims_lp

        return lp, Kinit_dim
