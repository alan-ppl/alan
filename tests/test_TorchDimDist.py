import torch as t
import torch.distributions as td
import functorch.dim
from functorch.dim import Dim, dims

from alan_simplified.TorchDimDist import TorchDimDist, tdd_order, generic_tdd_order

import unittest

def generic_tensor_eq(a, b):
    if a.shape != b.shape:
        return False

    if isinstance(a, functorch.dim.Tensor) and isinstance(b, functorch.dim.Tensor):
        return t.all(a==b).item() and a.dims == b.dims
    elif isinstance(a, t.Tensor) and  isinstance(b, t.Tensor):
        return t.all(a==b).item()
    else:
        return False

class TestTorchDimDist_Sample(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

        self.d3 = Dim('d3', 3)
        self.d4 = Dim('d4', 4)
        self.d5 = Dim('d5', 5)

        self.loc = t.randn((3,4,5))[self.d3, self.d4, self.d5]
        self.scale = t.randn((3,4,5))[self.d3, self.d4, self.d5].exp()

        self.tdd = TorchDimDist(td.Normal, loc=self.loc, scale=self.scale)

    def test_sample_reparam_true(self):
        sample = self.tdd.sample(True, [self.d3, self.d4, self.d5], [3,4])
        self.assertEqual(set(sample.dims), {self.d3, self.d4, self.d5})
        self.assertEqual(sample.shape, t.Size([3,4]))

    def test_sample_reparam_false(self):
        sample = self.tdd.sample(False, [self.d3, self.d4, self.d5], [3,4])
        self.assertEqual(set(sample.dims), {self.d3, self.d4, self.d5})
        self.assertEqual(sample.shape, t.Size([3,4]))

    def test_sample_with_extra_dims(self):
        d6 = Dim('d6', 6)
        sample = self.tdd.sample(True, [self.d3, self.d4, self.d5, d6], [3,4])

        self.assertEqual(set(sample.dims), {self.d3, self.d4, self.d5, d6})
        self.assertEqual(sample.shape, t.Size([3,4]))

    def test_sample_with_missing_dims(self):
        with self.assertRaises(AssertionError):
            self.tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_sample_with_reparam_not_supported(self):
        tdd = TorchDimDist(td.Poisson, rate=self.loc)
        with self.assertRaises(Exception):
            tdd.sample(True, [self.d3, self.d4, self.d5], [3,4])

    def test_incorrect_distribution(self):
        with self.assertRaises(Exception):
            TorchDimDist("Not a distribution", loc=self.loc, scale=self.scale)

    def test_negative_scale(self):
        negative_scale = -t.ones((3,4))[self.d3, self.d4]
        with self.assertRaises(Exception):
            tdd = TorchDimDist(td.Normal, loc=self.loc, scale=negative_scale)
            tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_zero_scale(self):
        zero_scale = t.zeros((3,4))[self.d3, self.d4]
        with self.assertRaises(Exception):
            tdd = TorchDimDist(td.Normal, loc=self.loc, scale=zero_scale)
            tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_infinite_loc(self):
        infinite_loc = t.ones((3,4))[self.d3, self.d4] * float('inf')
        with self.assertRaises(Exception):
            tdd = TorchDimDist(td.Normal, loc=infinite_loc, scale=self.scale)
            tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_nan_loc(self):
        nan_loc = t.ones((3,4))[self.d3, self.d4] * float('nan')
        with self.assertRaises(Exception):
            tdd = TorchDimDist(td.Normal, loc=nan_loc, scale=self.scale)
            tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_infinite_scale(self):
        infinite_scale = t.ones((3,4))[self.d3, self.d4] * float('inf')
        with self.assertRaises(Exception):
            tdd = TorchDimDist(td.Normal, loc=self.loc, scale=infinite_scale)
            tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_nan_scale(self):
        nan_scale = t.ones((3,4))[self.d3, self.d4] * float('nan')
        with self.assertRaises(Exception):
            tdd = TorchDimDist(td.Normal, loc=self.loc, scale=nan_scale)
            tdd.sample(True, [self.d3, self.d4], [3,4])

    def test_rand_dims(self):
        for seed in range(20):
            t.manual_seed(seed)

            num_dims = t.randint(1,10, ()).item()
            dim_sizes = t.randint(1, 10, (num_dims,))

            dim_objs = []
            for i in range(num_dims):
                d = Dim(f'd{i}', dim_sizes[i].item())
                dim_objs.append(d)

            tdd = TorchDimDist(td.Normal, loc=t.randn(tuple(dim_sizes))[dim_objs], scale=t.randn(tuple(dim_sizes))[dim_objs].exp())

            num_sample_dims = t.randint(1,5,()).item()
            sample_shape = t.Size(t.randint(1, 20, (num_sample_dims,)))

            sample = tdd.sample(True, dim_objs, sample_shape)

            self.assertEqual(set(sample.dims), set(dim_objs))
            self.assertEqual(sample.shape, sample_shape)

class TestTorchDimDist_tdd_order(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

        self.d3 = Dim('d3', 3)
        self.d4 = Dim('d4', 4)
        self.d5 = Dim('d5', 5)

    def test_tdd_order(self):
        dim_objs = [self.d3, self.d4, self.d5]
        tensor = t.randn((3,4,5,6,7,8))[dim_objs]
        
        self.assertEqual(set(tensor.dims), set(dim_objs))
        self.assertEqual(tensor.shape, t.Size([6,7,8]))

        # remove one named dim (d4)
        tensor_ordered = tdd_order(tensor, [self.d4], 0)
        self.assertEqual(set(tensor_ordered.dims), {self.d3, self.d5})
        self.assertEqual(tensor_ordered.shape, t.Size([6,7,8,4]))

        tensor_ordered = tdd_order(tensor, [self.d4], 1)
        self.assertEqual(set(tensor_ordered.dims), {self.d3, self.d5})
        self.assertEqual(tensor_ordered.shape, t.Size([6,7,4,8]))  # 8 is the event_ndim

        tensor_ordered = tdd_order(tensor, [self.d4], 2)
        self.assertEqual(set(tensor_ordered.dims), {self.d3, self.d5})
        self.assertEqual(tensor_ordered.shape, t.Size([6,4,7,8]))  # 7 and 8 are the event_ndims

    
        # remove two named dims (d3 and d4)
        tensor_ordered = tdd_order(tensor, [self.d3, self.d4], 0)
        self.assertEqual(set(tensor_ordered.dims), {self.d5})
        self.assertEqual(tensor_ordered.shape, t.Size([6,7,8,3,4]))

        tensor_ordered = tdd_order(tensor, [self.d3, self.d4], 1)
        self.assertEqual(set(tensor_ordered.dims), {self.d5})
        self.assertEqual(tensor_ordered.shape, t.Size([6,7,3,4,8]))  # 8 is the event_ndim

        tensor_ordered = tdd_order(tensor, [self.d3, self.d4], 2)
        self.assertEqual(set(tensor_ordered.dims), {self.d5})
        self.assertEqual(tensor_ordered.shape, t.Size([6,3,4,7,8]))  # 7 and 8 are the event_ndims


        # remove all three named dims (d3, d4 and d5)
        tensor_ordered = tdd_order(tensor, [self.d3, self.d4, self.d5], 0)
        with self.assertRaises(AttributeError):
            set(tensor_ordered.dims)
        self.assertEqual(tensor_ordered.shape, t.Size([6,7,8,3,4,5]))

        tensor_ordered = tdd_order(tensor, [self.d3, self.d4, self.d5], 1)
        with self.assertRaises(AttributeError):
            set(tensor_ordered.dims)        
        self.assertEqual(tensor_ordered.shape, t.Size([6,7,3,4,5,8]))  # 8 is the event_ndim

        tensor_ordered = tdd_order(tensor, [self.d3, self.d4, self.d5], 2)
        with self.assertRaises(AttributeError):
            set(tensor_ordered.dims)
        self.assertEqual(tensor_ordered.shape, t.Size([6,3,4,5,7,8]))  # 7 and 8 are the event_ndims

    def test_generic_tdd_order_with_dims(self):
        dim_objs = [self.d3, self.d4, self.d5]
        tensor = t.randn((3,4,5,6,7,8))
        
        named_tensor = tensor[dim_objs]

        named_tensor_ordered = tdd_order(named_tensor, dim_objs, 0)
        named_tensor_generic_ordered = generic_tdd_order(named_tensor, dim_objs, 0)
        self.assertTrue(generic_tensor_eq(named_tensor_ordered, named_tensor_generic_ordered))
        self.assertTrue(generic_tensor_eq(tensor.permute(3,4,5,0,1,2), named_tensor_ordered))
        self.assertTrue(generic_tensor_eq(tensor.permute(3,4,5,0,1,2), named_tensor_generic_ordered))

        named_tensor_ordered = tdd_order(named_tensor, dim_objs, 1)
        named_tensor_generic_ordered = generic_tdd_order(named_tensor, dim_objs, 1)
        self.assertTrue(generic_tensor_eq(named_tensor_ordered, named_tensor_generic_ordered))
        
        self.assertFalse(generic_tensor_eq(tensor, named_tensor_ordered))
        self.assertFalse(generic_tensor_eq(tensor, named_tensor_generic_ordered))

        self.assertTrue(generic_tensor_eq(tensor.permute(3,4,0,1,2,5), named_tensor_ordered))
        self.assertTrue(generic_tensor_eq(tensor.permute(3,4,0,1,2,5), named_tensor_generic_ordered))

        named_tensor_ordered = tdd_order(named_tensor, dim_objs, 2)
        named_tensor_generic_ordered = generic_tdd_order(named_tensor, dim_objs, 2)
        self.assertTrue(generic_tensor_eq(named_tensor_ordered, named_tensor_generic_ordered))
        
        self.assertFalse(generic_tensor_eq(tensor, named_tensor_ordered))
        self.assertFalse(generic_tensor_eq(tensor, named_tensor_generic_ordered))

        self.assertTrue(generic_tensor_eq(tensor.permute(3,0,1,2,4,5), named_tensor_ordered))
        self.assertTrue(generic_tensor_eq(tensor.permute(3,0,1,2,4,5), named_tensor_generic_ordered))

    def test_generic_tdd_order_no_dims(self):
        dim_objs = [self.d3, self.d4, self.d5]
        tensor = t.randn((3,4,5,6,7,8))
        
        tensor_generic_ordered = generic_tdd_order(tensor, dim_objs, 0)
        self.assertTrue(generic_tensor_eq(tensor, tensor_generic_ordered))

        tensor_generic_ordered = generic_tdd_order(tensor, dim_objs, 1)
        self.assertTrue(generic_tensor_eq(tensor, tensor_generic_ordered))

        tensor_generic_ordered = generic_tdd_order(tensor, dim_objs, 2)
        self.assertTrue(generic_tensor_eq(tensor, tensor_generic_ordered))


class TestTorchDimDist_log_prob(unittest.TestCase):
    def setUp(self):
        t.manual_seed(127)

        self.d3 = Dim('d3', 3)
        self.d4 = Dim('d4', 4)
        self.d5 = Dim('d5', 5)

        self.dims = [self.d3, self.d4, self.d5]

        self.loc = t.randn((3,4,5))[self.d3, self.d4, self.d5]
        self.scale = t.randn((3,4,5))[self.d3, self.d4, self.d5].exp()

        self.tdd = TorchDimDist(td.Normal, loc=self.loc, scale=self.scale)

    def test_log_prob_from_sample(self):
        sample_shape = [3,4]

        t.manual_seed(1)
        sample_tdd = self.tdd.sample(True, [self.d3, self.d4, self.d5], sample_shape)
        lp = self.tdd.log_prob(sample_tdd)

        lp_no_dims = lp.order(self.d3, self.d4, self.d5)

        t.manual_seed(1)
        regular_td = td.Normal(self.loc.order(*self.dims), self.scale.order(*self.dims))
        sample_td = regular_td.sample(sample_shape)
        lp_td = regular_td.log_prob(sample_td)

        unnamed_dims = range(len(sample_shape))
        torchdim_dims = range(len(sample_shape), len(sample_shape)+len(self.dims))

        lp_td = lp_td.permute(*torchdim_dims, *unnamed_dims)

        self.assertEqual(lp.shape, t.Size(sample_shape))
        self.assertTrue((lp_td == lp_no_dims).all())

    def test_log_prob_from_sample_with_event_dims(self):
        # self.mvn_loc = t.randn((3,4,5))[self.d3, self.d4]
        # self.mvn_cov_matrix = t.randn((3,4,5,5))[self.d3, self.d4].exp()

        # self.tdd_with_event_dims = TorchDimDist(td.MultivariateNormal, loc=self.mvn_loc, covariance_matrix=self.mvn_cov_matrix)

        # sample_shape = [3,4]

        # t.manual_seed(1)
        # sample_tdd = self.tdd.sample(True, [self.d3, self.d4], sample_shape)
        # lp = self.tdd.log_prob(sample_tdd)

        # lp_no_dims = lp.order(self.d3, self.d4)

        # t.manual_seed(1)
        # sample_shape = [3,4,3,4]
        # td_obj = td.MultivariateNormal(self.mvn_loc.order(*self.dims), self.mvn_cov_matrix.order(*self.dims))
        # sample_td = td_obj.sample(sample_shape)
        # lp_td = td_obj.log_prob(sample_td)

        # unnamed_dims = range(len(sample_shape))
        # torchdim_dims = range(len(sample_shape), len(sample_shape)+len(self.dims))

        # lp_td = lp_td.permute(*torchdim_dims, *unnamed_dims)

        # self.assertEqual(lp.shape, t.Size(sample_shape))
        # self.assertTrue((lp_td == lp_no_dims).all())

        pass


if __name__ == '__main__':
    unittest.main()