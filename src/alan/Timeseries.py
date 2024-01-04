from .dist import _Dist, sample_gdt
from .utils import *
from .Sampler import Sampler





#Checks to Timeseries init.
#if 0 == len(active_platedims):
#    raise Exception(f"Timeseries can't be in the top-layer plate, as there's no platesize at the top")
#if name not in self.trans.all_args:
#    raise Exception(f"The timeseries transition distribution for {name} must have some dependence on the previous timestep; you get that by including {name} as an argument in the transition distribution.")


class Timeseries:
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
        self.is_timeseries = True

        if not isinstance(init, str):
            raise Exception(f"the first / `init` argument in a Timeseries should be a string, representing a variable name in the above plate")

        if not isinstance(trans, _Dist):
            raise Exception("the second / `trans` argument in a Timeseries should be a distribution")

        if trans.sample_shape != t.Size([]):
            raise Exception("sample_shape on the transition distribution must not be set; if you want a sample_shape, it needs to be on the initial state")

        self.init = init
        self.trans = trans.finalize(None)
        #Will include own name, but that'll be eliminated in the first step of sample_gdt
        self.all_args = [init, *self.trans.all_args] 

    def sample(
            self,
            name:str,
            scope: dict[str, Tensor], 
            inputs_params: dict,
            active_platedims:list[Dim],
            all_platedims:dict[str, Dim],
            groupvarname2Kdim:dict[str, Dim],
            sampler:Sampler,
            reparam:bool,
            ):

        return sample_gdt(
            prog={name: self},
            scope=scope,
            K_dim=groupvarname2Kdim[name],
            groupvarname2Kdim=groupvarname2Kdim,
            active_platedims=active_platedims,
            sampler=sampler,
            reparam=reparam,
        )[name]
#    def sample(
#            self,
#            name:str,
#            scope: dict[str, Tensor], 
#            inputs_params: dict,
#            active_platedims:list[Dim],
#            all_platedims:dict[str, Dim],
#            groupvarname2Kdim:dict[str, Dim],
#            sampler:Sampler,
#            reparam:bool,
#            ):
#
#        if name not in self.trans.all_args:
#            raise Exception(f"The timeseries transition distribution for {name} must have some dependence on the previous timestep; you get that by including {name} as an argument in the transition distribution.")
#
#        if 0 == len(active_platedims):
#            raise Exception(f"Timeseries can't be in the top-layer plate, as there's no platesize at the top")
#
#        active_platedims, T_dim = (active_platedims[:-1], active_platedims[-1])
#        K_dim = groupvarname2Kdim[name]
#        Kprev_dim = Dim(f"{K_dim}_prev", K_dim.size)
#        sample_dims = [K_dim, *active_platedims] #Don't sample T_dim.
#
#        if self.init not in scope:
#            raise Exception(f"Initial state, {self.init}, not in scope for timeseries {name}")
#
#        #Set previous state equal to initial state.
#        prev_state = scope[self.init]
#        #Make sure K-dimension for initial state is Kprev_dim.
#        prev_state = prev_state.order(groupvarname2Kdim[self.init])[Kprev_dim]
#        #Check that prev_state has the right dimensions
#        if set(prev_state.dims) != set([Kprev_dim, *active_platedims]):
#            raise Exception(f"Initial state, {self.init}, doesn't have the right dimensions for timeseries {name}; the initial state must be defined one step up in the plate heirarchy")
#
#        unresampled_scope = self.filter_scope(scope) #drops initial state.
#
#        results = []
#
#        for time in range(T_dim.size):
#            scope = {}
#
#            #select out the time'th element of anything with a time-dimension.
#            for k, v in unresampled_scope.items():
#                if T_dim in set(generic_dims(v)):
#                    v = v.order(T_dim)[time]
#                scope[k] = v
#
#            scope[name] = prev_state
#            scope = sampler.resample_scope(scope, active_platedims, K_dim)
#
#            tdd = self.trans.tdd(scope)
#            sample = tdd.sample(reparam, sample_dims, sample_shape=t.Size([]))
#
#            results.append(sample)
#            prev_state = sample.order(K_dim)[Kprev_dim]
#
#        return t.stack(results, 0)[T_dim]
#