from .dist import _Dist
from .utils import *
from .Sampler import Sampler

class Timeseries:
    """
    Represents a timeseries.

    init and trans are distributions, e.g.:

    .. code-block:: python

       Plate(
           ts_init = Normal(0., 1.),
           T = Plate(
               ts = Timeseries('ts_init', Normal(lambda ts: 0.9*ts, 0.1)),
           )
       )
    Here
    * ``T`` is the plate (i.e. ``all_platesizes['T']``) is the length of the timeseries.  Note that this is a slight abuse of the term "Plate", which is usually only used to refer to independent variables.
    * ``ts`` is the name of the timeseries random variable itself.
    * ``Normal(lambda ts: 0.9*ts, 0.1)`` is the transition distribution.  Note that it refers to the previous step of itself using the timeseries name itself, ``ts``, as an argument.
    * ``ts_init`` is the initial state. Must be a string representing a random variable in the previous plate.
    * Implementation notes:
      - Stick the previous timestep into scope, converting K_ts into K_ts_prev.
      - sample returns K_ts x T tensor, obtained by sampling T times from the transition distribution.
      - log_PQ for Timeseries:
        - Timeseries.trans_log_PQ returns a K_ts_prev x K_ts x T tensor, where:
          - K_ts represents timestep t
          - K_ts_prev represents timestep t-1
      - log_PQ_plate:
        - Non-split log_PQ_plate returns a K_ts_init tensor.
        - Splitting log_PQ_plate:
          - Uses a backward pass, so at the start of the backward pass, we sum from the back.
          - At the start of the backward pass, log_PQ_plate takes one unusual input: initial timeseries state, with dimension K_ts.  If initial timeseries state is provided as a kwarg, we ignore Timeseries.init.
          - log_PQ_plate returns K_ts dimensional tensor, resulting from summing all the way from the back to the start of the split.
          - The next split takes two unusual arguments: the initial state, and the log_pq from the last split evaluated by the backward pass.
        - Algorithm:
          - Inputs: 
            - optional initial timeseries state (K_inits) (only provided for splits, if not provided, it this is computed from Timeseries.init).
            - samples (all with T timesteps).
            - final log_PQ with K_ts dimension.
          - gather all log-probs, which must all have T timesteps so that reduce_Ks is easy.
            - for timeseries, having all log_prob tensors with length T isn't trivial.
            - to have T timesteps, we must have T+1 states: first and rest.
            - to ensure that the log-probs for all the transitions have the same shape (which they need to to ensure that we can put them together into a single length-T tensor, we need the first state to have the same K-dimension as the rest.)
            - that's why first state must be a random variable in the previous plate.
          - Then, we reduce_Ks, keeping K_ts_prev, K_ts, 
          - Then we reduce over time using chain_matmul.  Ultimately, 

    Note:
       OptParam and QEMParam are currently banned in timeseries.
    """
    def __init__(self, init, trans):

        if not isinstance(init, str):
            raise Exception(f"the first / `init` argument in a Timeseries should be a string, representing a variable name in the above plate")

        if not isinstance(trans, _Dist):
            raise Exception("the second / `trans` argument in a Timeseries should be a distribution")

        if trans.sample_shape != t.Size([]):
            raise Exception("sample_shape on the transition distribution must be default value; if you want a sample_shape, it needs to be on the initial state")

        self.init = init
        self.trans = trans.finalize(varname=None) #varname=None raises exception when we use OptParam / QEMParam

    def filter_scope(self, scope: dict[str, Tensor]):
        return {k: v for (k,v) in scope.items() if k in self.trans.all_args}

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

        active_platedims, T_dim = (active_platedims[:-1], active_platedims[-1])
        K_dim = groupvarname2Kdim[name]
        Kprev_dim = Dim(f"{K_dim}_prev", K_dim.size)
        sample_dims = [K_dim, *active_platedims] #Don't sample T_dim.

        if self.init not in scope:
            raise Exception(f"Initial state, {self.init}, not in scope for timeseries {name}")

        #Set previous state equal to initial state.
        prev_state = scope[self.init]
        #Make sure K-dimension for initial state is Kprev_dim.
        prev_state = prev_state.order(groupvarname2Kdim[self.init])[Kprev_dim]
        #Check that prev_state has the right dimensions
        if set(prev_state.dims) != set([Kprev_dim, *active_platedims]):
            raise Exception(f"Initial state, {self.init}, doesn't have the right dimensions for timeseries {name}; the initial state must be defined one step up in the plate heirarchy")

        unresampled_scope = self.filter_scope(scope) #drops initial state.

        results = []

        for _ in range(T_dim.size):
            scope = {**unresampled_scope}
            scope[name] = prev_state
            scope = sampler.resample_scope(scope, active_platedims, K_dim)

            tdd = self.trans.tdd(scope)
            sample = tdd.sample(reparam, sample_dims, sample_shape=t.Size([]))

            results.append(sample)
            prev_state = sample.order(K_dim)[Kprev_dim]


        return t.stack(results, 0)[T_dim]





