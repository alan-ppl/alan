import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries

plate = Plate( 
    ts1_init = Normal(0., 1.),
    ts2_init = Normal(0., 1.),
    T = Plate(
        ts1 = Timeseries('ts1_init', Normal(lambda ts1: 0.9*ts1, 0.1)),
        ts2 = Timeseries('ts2_init', Normal(lambda ts1, ts2: 0.9*ts1 + ts2, 0.1)),
        a = Normal('ts2', 1.)
    ),
)

bound_plate = BoundPlate(plate, {'T': 3})
sample = bound_plate.sample()
