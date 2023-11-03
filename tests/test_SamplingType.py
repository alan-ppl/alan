import torch as t
import torch.distributions as td
from functorch.dim import Dim

from alan_simplified import Normal, Plate, BoundPlate, SingleSample, Problem
from alan_simplified.SamplingType import *

t.manual_seed(0)

Kdim = Dim('K')
parent_1_Kdim = Dim('parent_1_K')
parent_2_Kdim = Dim('parent_2_K')
plate_1 = Dim('plate_1')
plate_2 = Dim('plate_2')

class test_SingleSample():
    
    def __init__(self):
        self.scope  = {'a':t.randn(6)[plate_1], 'b':t.randn(6,7)[plate_1,plate_2]}
        
        self.sample = SingleSample()
        
        self.lp = t.randn(6)[plate_1]
        
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, None)
        assert scope == self.scope
    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_log_prob(self.lp, None, {}, [])
        assert lp == self.lp
        
class test_Parallel():
    
    def __init__(self):
        self.scope  = {'a':t.randn(5,6)[Kdim,plate_1], 'b':t.randn(5,6,7)[Kdim,plate_1,plate_2]}
        
        self.sample = Parallel()
        
        self.lp = t.randn(5,5,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1]
        
        self.varname2Kdim = {'parent_1_K':parent_1_Kdim, 'parent_2_K':parent_2_Kdim, 'a':Kdim}
        
        # self.after_lp = t.randn(5,6)[Kdim, plate_1]
    
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, None)
        assert scope == self.scope
    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_log_prob(self.lp, 'a', self.varname2Kdim, [plate_1])
        assert lp.shape == []
        assert set(lp.dims) == {Kdim, plate_1}
        # assert lp == self.after_lp
        
class test_MixturePermutation():
    
    def __init__(self):
        self.scope  = {'a':t.randn(5,6)[Kdim,plate_1], 'b':t.randn(5,6,7)[Kdim,plate_1,plate_2]}
        
        self.scope_after = {} #Fill this in
        self.sample = MixturePermutation()
        
        self.lp = t.randn(5,5,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1]
        
        self.varname2Kdim = {'parent_1_K':parent_1_Kdim, 'parent_2_K':parent_2_Kdim, 'a':Kdim}
        
        # self.after_lp = t.randn(5,6)[Kdim, plate_1]
    
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, None)
        assert scope.keys() == self.scope.keys()
        assert scope['a'].shape == self.scope['a'].shape
        assert set(scope['a'].dims) == set(self.scope['a'].dims)
        assert scope['b'].shape == self.scope['b'].shape
        assert set(scope['b'].dims) == set(self.scope['b'].dims)
    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_log_prob(self.lp, 'a', self.varname2Kdim, [plate_1])
        assert lp.shape == []
        assert set(lp.dims) == {Kdim, plate_1}
        # assert lp == self.after_lp
        
class test_MixtureCategoricaln():
    
    def __init__(self):
        self.scope  = {'a':t.randn(5,6)[Kdim,plate_1], 'b':t.randn(5,6,7)[Kdim,plate_1,plate_2]}
        
        self.scope_after = {} #Fill this in
        self.sample = MixtureCategorical()
        
        self.lp = t.randn(5,5,5,6)[Kdim, parent_1_Kdim, parent_2_Kdim, plate_1]
        
        self.varname2Kdim = {'parent_1_K':parent_1_Kdim, 'parent_2_K':parent_2_Kdim, 'a':Kdim}
        
        # self.after_lp = t.randn(5,6)[Kdim, plate_1]
    
    def test_resample_scope(self):
        scope = self.sample.resample_scope(self.scope, None)
        assert scope.keys() == self.scope.keys()
        assert scope['a'].shape == self.scope['a'].shape
        assert set(scope['a'].dims) == set(self.scope['a'].dims)
        assert scope['b'].shape == self.scope['b'].shape
        assert set(scope['b'].dims) == set(self.scope['b'].dims)
    
    def test_reduce_log_prob(self):
        lp = self.sample.reduce_log_prob(self.lp, 'a', self.varname2Kdim, [plate_1])
        assert lp.shape == []
        assert set(lp.dims) == {Kdim, plate_1}
        # assert lp == self.after_lp