import torch as t
import functorch.dim
from functorch.dim import Dim, dims

from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample
from alan_simplified.IndexedSample import IndexedSample

import unittest

Tensor = (t.Tensor, functorch.dim.Tensor)

def generic_tensordict_eq(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False

        for key in a.keys():
            if not generic_tensordict_eq(a[key], b[key]):
                return False

        return True
    
    else:
        assert isinstance(a, Tensor) and isinstance(b, Tensor)
    
        if a.shape != b.shape:
            return False

        if isinstance(a, functorch.dim.Tensor) and isinstance(b, functorch.dim.Tensor):
            return t.all(a==b).item() and a.dims == b.dims
        elif isinstance(a, Tensor) and  isinstance(b, Tensor):
            return t.all(a==b).item()
        else:
            return False
        
def nested_tensor_dict_assert(d, condition):
    if isinstance(d, dict):
        for key, value in d.items():
            if not nested_tensor_dict_assert(value, condition):
                return False
            
        return True
    else:
        assert isinstance(d, Tensor)
        return condition(d)
    
def tensor_dict_structure_eq(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False

        for key in a.keys():
            if not tensor_dict_structure_eq(a[key], b[key]):
                return False

        return True

    else:
        assert isinstance(a, Tensor) and isinstance(b, Tensor)

class TestIndexedSample_index_in(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

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
                ),
            ),
        )
        Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

        self.platesizes = {'p1': 3, 'p2': 4}
        self.data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

        self.prob = Problem(P, Q, self.platesizes, self.data)

        sampling_type = IndependentSample
        self.sample = self.prob.sample(3, True, sampling_type)

        self.post_idxs = self.sample.sample_posterior(num_samples=10)

    def test_index_in(self):
        isample = IndexedSample(self.sample, self.post_idxs)

        # Check isample.sample has Ndims rather than Kdims
        nested_tensor_dict_assert(isample.sample, lambda x: isinstance(x, Tensor))
        nested_tensor_dict_assert(isample.sample, lambda x: isample.Ndim in set(x.dims))
        nested_tensor_dict_assert(isample.sample, lambda x: not any([Kdim in set(x.dims) for Kdim in self.post_idxs.keys()]))

        # Check isample.sample has the same structure as self.sample.sample
        tensor_dict_structure_eq(isample.sample, self.sample.sample)
        

class TestIndexedSample_predictive(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

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
                ),
            ),
        )
        Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

        self.platesizes = {'p1': 3, 'p2': 4}
        self.data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

        self.prob = Problem(P, Q, self.platesizes, self.data)

        sampling_type = IndependentSample
        self.sample = self.prob.sample(3, True, sampling_type)

        self.post_idxs = self.sample.sample_posterior(num_samples=10)

        self.isample = IndexedSample(self.sample, self.post_idxs)

    def test_predictive_no_extended_data(self):
        extended_platesizes = {'p1': 5, 'p2': 6}

        predictive_samples, ll_train, ll_all = self.isample._predictive(self.prob.P, extended_platesizes, True, None)

        assert ll_train == {}
        assert ll_all == {}

        # Check predictive_samples has Ndims rather than Kdims
        nested_tensor_dict_assert(predictive_samples, lambda x: isinstance(x, Tensor))
        nested_tensor_dict_assert(predictive_samples, lambda x: self.isample.Ndim in set(x.dims))
        nested_tensor_dict_assert(predictive_samples, lambda x: not any([Kdim in set(x.dims) for Kdim in self.post_idxs.keys()]))

        # Check predictive_samples has the same structure as self.sample.sample
        tensor_dict_structure_eq(predictive_samples, self.sample.sample)

    def test_predictive_with_extended_data(self):
        extended_platesizes = {'p1': 5, 'p2': 6}
        extended_data = {'e': t.randn(5, 6, names=('p1', 'p2'))}

        predictive_samples, ll_train, ll_all = self.isample._predictive(self.prob.P, extended_platesizes, True, extended_data)

        nested_tensor_dict_assert(predictive_samples, lambda x: isinstance(x, Tensor))

        # Check predictive_samples has Ndims rather than Kdims
        # (First take out data which won't have an Ndim)
        pred_samples_no_data = predictive_samples.copy()
        pred_samples_no_data['p1']['p2'] = {}
        nested_tensor_dict_assert(pred_samples_no_data, lambda x: not any([Kdim in set(x.dims) for Kdim in self.post_idxs.keys()]))
        nested_tensor_dict_assert(pred_samples_no_data, lambda x: self.isample.Ndim in set(x.dims))

        # Check predictive_samples has the same structure as self.sample.sample
        tensor_dict_structure_eq(predictive_samples, self.sample.sample)

        # Check ll_train and ll_all contain the same variables as self.data
        assert set(ll_train.keys()) == set(ll_all.keys())
        assert set(ll_train.keys()) == set(self.data.keys())

        # Check ll_train and ll_all contain a singleton tensor for each variable
        for key in ll_train.keys():
            assert isinstance(ll_train[key], Tensor)
            assert isinstance(ll_all[key], Tensor)

            assert ll_train[key].shape == ()
            assert ll_all[key].shape == ()  

    def test_predictive_sample(self):
        extended_platesizes = {'p1': 5, 'p2': 6}

        predictive_samples = self.isample.predictive_sample(self.prob.P, extended_platesizes, True)

        # Check predictive_samples has Ndims rather than Kdims
        nested_tensor_dict_assert(predictive_samples, lambda x: isinstance(x, Tensor))
        nested_tensor_dict_assert(predictive_samples, lambda x: self.isample.Ndim in set(x.dims))
        nested_tensor_dict_assert(predictive_samples, lambda x: not any([Kdim in set(x.dims) for Kdim in self.post_idxs.keys()]))

        # Check predictive_samples has the same structure as self.sample.sample
        tensor_dict_structure_eq(predictive_samples, self.sample.sample)

    def test_predictive_ll(self):
        extended_platesizes = {'p1': 5, 'p2': 6}
        extended_data = {'e': t.randn(5, 6, names=('p1', 'p2'))}

        ll = self.isample.predictive_ll(self.prob.P, extended_platesizes, True, extended_data)

        # Check ll_train and ll_all contain the same variables as self.data
        assert set(ll.keys()) == set(self.data.keys())

        # Check ll_train and ll_all contain a singleton tensor for each variable
        for key, value in ll.items():
            assert isinstance(value, Tensor)

            assert value.shape == ()


if __name__ == '__main__':
    unittest.main()