import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Plate, BoundPlate, SingleSample, Problem

import unittest

class TestElbo(unittest.TestCase):

    def test_elbo(self):
        P = Plate(
            a = Normal(0, 1),
            b = Normal("a", 1),
            c = Normal("b", lambda a: a.exp()),
        )
        Q = Plate(
            a = Normal("a_loc", lambda log_a_scale: log_a_scale.exp()),
            b = Normal("b_loc", lambda log_b_scale: log_b_scale.exp())
        )
        Q = BoundPlate(Q, parameters={
            "a_loc"       : t.zeros(()),
            "log_a_scale" : t.zeros(()),
            "b_loc"       : t.zeros(()),
            "log_b_scale" : t.zeros(())
        })

        problem = Problem(P, Q, all_platesizes={}, data={'c': 5*t.ones(())})
        sample = problem.sample(K=10, reparam=True)
        L = sample.elbo()
        print(L)

if __name__ == '__main__':
    unittest.main()