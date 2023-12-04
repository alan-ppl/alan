import torch as t
import functorch.dim
from functorch.dim import Dim, dims

from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample
from alan_simplified.reduce_Ks import reduce_Ks, sample_Ks, logsumexp_sum

import unittest

Tensor = (t.Tensor, functorch.dim.Tensor)
        

class TestReduceKs(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

        self.Kdim = Dim('K')

        self.parent_1_Kdim = Dim('parent_1_K')
        self.parent_2_Kdim = Dim('parent_2_K')
        self.plate_1 = Dim('plate_1')
        self.plate_2 = Dim('plate_2')

        self.sizes = {'K': 2, 'parent_1_K': 3, 'parent_2_K': 4}

    def test_reduce_Ks(self):
        lps = [t.randn(2,3,4,5)[self.Kdim, self.parent_1_Kdim, self.parent_2_Kdim, self.plate_1],
               t.randn(2,3,5)[self.Kdim, self.parent_1_Kdim, self.plate_1], 
               t.randn(2,4,5)[self.Kdim, self.parent_2_Kdim, self.plate_1]]
        
        reduced = reduce_Ks(lps, [self.Kdim, self.parent_1_Kdim, self.parent_2_Kdim])

        assert set(reduced.dims) == {self.plate_1}

        
class TestSampleKs(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

        self.Kdim = Dim('K')


        self.parent_1_Kdim = Dim('parent_1_K')
        self.parent_2_Kdim = Dim('parent_2_K')
        self.plate_1 = Dim('plate_1')
        self.plate_2 = Dim('plate_2')

        self.sizes = {'K': 2, 'parent_1_K': 3, 'parent_2_K': 4}

        self.Ndim = Dim('N')


    def test_sample_Ks_single_plate(self):
        lps = [t.randn(2,3,4,5)[self.Kdim, self.parent_1_Kdim, self.parent_2_Kdim, self.plate_1], 
               t.randn(2,3,5)[self.Kdim, self.parent_1_Kdim, self.plate_1], 
               t.randn(2,4,5)[self.Kdim, self.parent_2_Kdim, self.plate_1]]

        samples = sample_Ks(lps, [self.Kdim, self.parent_1_Kdim, self.parent_2_Kdim], self.Ndim, num_samples=10)

        for k,v in samples.items():
            assert set(v.dims) == {self.Ndim, self.plate_1}
            
    def test_sample_Ks_double_plates(self):

        lps = [t.randn(2,3,4,5,6)[self.Kdim, self.parent_1_Kdim, self.parent_2_Kdim, self.plate_1, self.plate_2], 
               t.randn(2,3,5,6)[self.Kdim, self.parent_1_Kdim, self.plate_1, self.plate_2], 
               t.randn(2,4,5,6)[self.Kdim, self.parent_2_Kdim, self.plate_1, self.plate_2]]

        samples = sample_Ks(lps, [self.Kdim, self.parent_1_Kdim, self.parent_2_Kdim], self.Ndim, num_samples=10)

        for k,v in samples.items():
            assert set(v.dims) == {self.Ndim, self.plate_1, self.plate_2}
    
class Testlogsumexp_sum(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)


    def test_logsumexp_sum(self):
        pass


if __name__ == '__main__':
    unittest.main()