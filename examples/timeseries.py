import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Timeseries

plate = Plate( 
    ts_init = Normal(0., 1.),
    T = Plate(
        ts = Timeseries('ts_init', Normal(lambda ts: 0.9*ts, 0.1)),
        a = Normal('ts', 1.)
    ),
)

bound_plate = BoundPlate(plate, {'T': 3})
sample = bound_plate.sample()
