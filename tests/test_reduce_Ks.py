import torch as t
import functorch.dim
from functorch.dim import Dim, dims


from alan_simplified import Normal, Bernoulli, Plate, BoundPlate, Group, Problem, IndependentSample, Data
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
            
    def test_Sample_Ks_compare_conditionals(self):
        P = Plate(
            ab = Group(
                a = Normal(0, 1),
                b = Normal("a", 1),
            ),
            c = Normal(0, lambda a: a.exp()),
            p1 = Plate(
                d = Normal("a", 1),
                p2 = Plate(
                    e = Normal("d", 1.),
                ),
            ),
        )

        Q = Plate(
            ab = Group(
                a = Normal("a_mean", 1),
                b = Normal("a", 1),
            ),
            c = Normal(0, lambda a: a.exp()),
            p1 = Plate(
                d = Normal("d_mean", 1),
                p2 = Plate(
                    e = Data()
                ),
            ),
        )
        Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

        all_platesizes = {'p1': 3, 'p2': 4}
        data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

        prob = Problem(P, Q, all_platesizes, data)

        sampling_type = IndependentSample
        sample = prob.sample(3, True, sampling_type)

        posterior_samples = list(sample.sample_posterior(num_samples=1000).values())

        marginals = sample.marginals()

        ab = posterior_samples[0].order(posterior_samples[0].dims) 
        c = posterior_samples[1].order(posterior_samples[1].dims)
        d = posterior_samples[2].order(posterior_samples[2].dims[1])

        posterior_ab = t.bincount(ab) / 1000
        source_term_ab = marginals['ab'].rename(None)
        assert (t.abs(posterior_ab - source_term_ab) < 0.05).all()

        posterior_c = t.bincount(c) / 1000
        print(marginals['c'])
        source_term_c = marginals['c'].rename(None)
        assert (t.abs(posterior_c - source_term_c) < 0.05).all()

        posterior_d = (t.bincount(d) / 1000).order(d.dims[0])
        source_term_d = marginals['d'].rename(None)
        assert (t.abs(posterior_d - source_term_d) < 0.05).all()
        
        
class Testlogsumexp_sum(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)


    def test_logsumexp_sum(self):
        pass


if __name__ == '__main__':
    unittest.main()