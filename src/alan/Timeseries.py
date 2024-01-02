from .dist import _Dist

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
        assert isinstance(init, str)
        assert isinstance(trans, _Dist)

        self.init = init
        self.trans = trans

