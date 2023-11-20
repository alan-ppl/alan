import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Plate, BoundPlate, Problem
from alan_simplified.SamplingType import *

import unittest

t.manual_seed(0)

Kdim = Dim('K')
parent_1_Kdim = Dim('parent_1_K')
parent_2_Kdim = Dim('parent_2_K')
plate_1 = Dim('plate_1')
plate_2 = Dim('plate_2')

# class test_SingleSample(unittest.TestCase):
    
#     def __init__(self):
#         self.scope  = {'a':t.randn(6)[plate_1], 'b':t.randn(6,7)[plate_1,plate_2]}
        
#         self.sample = SingleSample()
        
#         self.lp = t.randn(6)[plate_1]
        
#     def test_resample_scope(self):
#         scope = self.sample.resample_scope(self.scope, None)
#         assert scope == self.scope
    
#     def test_reduce_log_prob(self):
#         lp = self.sample.reduce_log_prob(self.lp, None, {}, [])
#         assert lp == self.lp
        
class test_Independent(unittest.TestCase):
    
    def setUp(self):
        self.scope  = {'a':t.randn(5,6)[parent_1_Kdim,plate_1], 'b':t.randn(5,6,7)[parent_2_Kdim,plate_1,plate_2]}
        self.scope_check = {'a':t.randn(5,6)[Kdim,plate_1], 'b':t.randn(5,6,7)[Kdim,plate_1,plate_2]}
        
        
        self.lp = t.randn(5,5,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1]
        
        varname2Kdim = {'parent_1_K':parent_1_Kdim, 'parent_2_K':parent_2_Kdim, 'a':Kdim}
        
        self.sample = IndependentSample()
        # self.after_lp = t.randn(5,6)[Kdim, plate_1]
    
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, [plate_1,plate_2], Kdim)

        for v1, v2 in zip(scope.values(), self.scope_check.values()):
            assert v1.shape == v2.shape
            # assert v1.dims == v2.dims

    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_logQ(self.lp, [plate_1], Kdim)

        assert lp.size() == t.Size([])
        assert set(lp.dims) == {Kdim, plate_1}
        # assert lp == self.after_lp
        
class test_MixturePermutation(unittest.TestCase):
    
    def setUp(self):
        self.scope  = {'a':t.randn(5,6)[parent_1_Kdim,plate_1], 'b':t.randn(5,6,7)[parent_2_Kdim,plate_1,plate_2]}
        self.scope_check = {'a':t.randn(5,6)[Kdim,plate_1], 'b':t.randn(5,6,7)[Kdim,plate_1,plate_2]}
        
        self.scope_after = {} #Fill this in
        self.sample = PermutationMixtureSample()
        
        self.lp = t.randn(5,5,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1]
        
        self.varname2Kdim = {'parent_1_K':parent_1_Kdim, 'parent_2_K':parent_2_Kdim, 'a':Kdim}
        
        # self.after_lp = t.randn(5,6)[Kdim, plate_1]
    
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, [plate_1,plate_2], Kdim)
        assert scope.keys() == self.scope_check.keys()
        assert scope['a'].shape == self.scope_check['a'].shape
        assert set(scope['a'].dims) == set(self.scope_check['a'].dims)
        assert scope['b'].shape == self.scope_check['b'].shape
        assert set(scope['b'].dims) == set(self.scope_check['b'].dims)
    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_logQ(self.lp, [plate_1], Kdim)
        assert lp.size() == t.Size([])
        assert set(lp.dims) == {Kdim, plate_1}
        # assert lp == self.after_lp
        
class test_MixtureCategorical(unittest.TestCase):
    
    def setUp(self):
        self.scope  = {'a':t.randn(5,6)[parent_1_Kdim,plate_1], 'b':t.randn(5,6,7)[parent_2_Kdim,plate_1,plate_2]}
        self.scope_check = {'a':t.randn(5,6)[Kdim,plate_1], 'b':t.randn(5,6,7)[Kdim,plate_1,plate_2]}
        
        self.scope_after = {} #Fill this in
        self.sample = CategoricalMixtureSample()
        
        self.lp = t.randn(5,5,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1]
        
        self.varname2Kdim = {'parent_1_K':parent_1_Kdim, 'parent_2_K':parent_2_Kdim, 'a':Kdim}
        
        # self.after_lp = t.randn(5,6)[Kdim, plate_1]
    
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, [plate_1,plate_2], Kdim)
        assert scope.keys() == self.scope_check.keys()
        assert scope['a'].shape == self.scope_check['a'].shape
        assert set(scope['a'].dims) == set(self.scope_check['a'].dims)
        assert scope['b'].shape == self.scope_check['b'].shape
        assert set(scope['b'].dims) == set(self.scope_check['b'].dims)
    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_logQ(self.lp, [plate_1], Kdim)
        assert lp.size() == t.Size([])
        assert set(lp.dims) == {Kdim, plate_1}
        # assert lp == self.after_lp
        
if __name__ == '__main__':
    unittest.main()