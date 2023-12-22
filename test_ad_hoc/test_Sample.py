import torch as t
import functorch.dim
from functorch.dim import Dim, dims

from alan_simplified import Normal, Plate, BoundPlate, Group, Problem, IndependentSample, Data


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

def mean(x):
    sample = x
    dim = x.dims[-1]

    w = 1/dim.size
    return (w * sample).sum(dim)

def second_moment(x):
    return mean(t.square(x))

def square(x):
    return x**2

def var(x):
    return mean(square(x)) - square(mean(x))

class TestSample_index_in(unittest.TestCase):

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
                    e = Data(),
                ),
            ),
        )
        P = BoundPlate(P)
        Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

        self.platesizes = {'p1': 3, 'p2': 4}
        self.data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

        self.prob = Problem(P, Q, self.platesizes, self.data)

        sampling_type = IndependentSample
        self.sample = self.prob.sample(3, True, sampling_type)

        self.importance_samples = self.sample.importance_samples(num_samples=10)
        self.post_idxs, _ = self.sample.importance_sampled_idxs(num_samples=10)

    def test_index_in(self):

        # Check importance_sample has Ndims rather than Kdims
        nested_tensor_dict_assert(self.importance_samples, lambda x: isinstance(x, Tensor))
        nested_tensor_dict_assert(self.importance_samples, lambda x: 'N' in set(x.names))
        nested_tensor_dict_assert(self.importance_samples, lambda x: not any([str(Kdim) in set(x.names) for Kdim in self.post_idxs.keys()]))

        # Check importance_sample has the same structure as self.sample.sample
        tensor_dict_structure_eq(self.importance_samples, self.sample.sample)
        

class TestSample_predictive(unittest.TestCase):

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
                    e = Data(),
                ),
            ),
        )
        P = BoundPlate(P)
        Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

        self.platesizes = {'p1': 3, 'p2': 4}
        self.data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

        self.prob = Problem(P, Q, self.platesizes, self.data)

        sampling_type = IndependentSample
        self.sample = self.prob.sample(3, True, sampling_type)
        self.post_idxs, _ = self.sample.importance_sampled_idxs(num_samples=10)



    def test_predictive_no_extended_data(self):
        extended_platesizes = {'p1': 5, 'p2': 6}

        predictive_samples, ll_train, ll_all, _ = self.sample._predictive(extended_platesizes, True, None, num_samples=10, all_inputs = {})

        assert ll_train == {}
        assert ll_all == {}

        # Check predictive_samples has Ndims rather than Kdims
        nested_tensor_dict_assert(predictive_samples, lambda x: isinstance(x, Tensor))
        nested_tensor_dict_assert(predictive_samples, lambda x: 'N' in set(x.names))
        nested_tensor_dict_assert(predictive_samples, lambda x: not any([str(Kdim) in set(x.names) for Kdim in self.post_idxs.keys()]))

        # Check predictive_samples has the same structure as self.sample.sample
        tensor_dict_structure_eq(predictive_samples, self.sample.sample)

    def test_predictive_with_extended_data(self):
        extended_platesizes = {'p1': 5, 'p2': 6}
        extended_data = {'e': t.randn(5, 6, names=('p1', 'p2'))}

        predictive_samples, ll_train, ll_all, _ = self.sample._predictive(extended_platesizes, True, extended_data, num_samples=10, all_inputs = {})

        nested_tensor_dict_assert(predictive_samples, lambda x: isinstance(x, Tensor))

        # Check predictive_samples has Ndims rather than Kdims
        # (First take out data which won't have an Ndim)
        pred_samples_no_data = predictive_samples.copy()
        pred_samples_no_data['p1']['p2'] = {}
        nested_tensor_dict_assert(predictive_samples, lambda x: 'N' in set(x.names))
        nested_tensor_dict_assert(predictive_samples, lambda x: not any([str(Kdim) in set(x.names) for Kdim in self.post_idxs.keys()]))

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

        predictive_samples = self.sample.predictive_sample(extended_platesizes, True, num_samples=10, all_inputs = {})

        # Check predictive_samples has Ndims rather than Kdims
        nested_tensor_dict_assert(predictive_samples, lambda x: isinstance(x, Tensor))
        nested_tensor_dict_assert(predictive_samples, lambda x: 'N' in set(x.names))
        nested_tensor_dict_assert(predictive_samples, lambda x: not any([str(Kdim) in set(x.names) for Kdim in self.post_idxs.keys()]))

        # Check predictive_samples has the same structure as self.sample.sample
        tensor_dict_structure_eq(predictive_samples, self.sample.sample)

    def test_predictive_ll(self):
        extended_platesizes = {'p1': 5, 'p2': 6}
        extended_data = {'e': t.randn(5, 6, names=('p1', 'p2'))}

        ll = self.sample.predictive_ll(extended_platesizes, True, extended_data, num_samples=10, all_inputs = {})

        # Check ll_train and ll_all contain the same variables as self.data
        assert set(ll.keys()) == set(self.data.keys())

        # Check ll_train and ll_all contain a singleton tensor for each variable
        for key, value in ll.items():
            assert isinstance(value, Tensor)

            assert value.shape == ()

class TestSample_predictive_analytic(unittest.TestCase):

    def setUp(self):
        t.manual_seed(127)

        self.data = t.tensor([0,1,2])
        self.extended_data = t.tensor([0,1,2,3])
        #Doing true pred_ll first because changing num_samples changes the result of predictive_ll if we do this after for some reason
        self.extended_platesizes = {'p1': 4}
        self.extended_data = {'obs': self.extended_data.refine_names('p1')} 


        posterior_mean = (3+1)**(-1) * (3*t.tensor([0.0,1.0,2.0]).mean() + 0)
        posterior_var = (3+1)**(-1)
        #By hand pred_ll
        pred_dist = t.distributions.Normal(posterior_mean, (1 + posterior_var)**(1/2))
        self.true_pred_lik = pred_dist.log_prob(t.tensor([3.0])).sum()


        self.sampling_type = IndependentSample

        self.P = Plate(mu = Normal(0, 1), 
                                p1 = Plate(obs = Normal("mu", 1)))
                
        self.Q = Plate(mu = Normal("mu_mean", 1),
                        p1 = Plate(obs = Data()))

        self.P = BoundPlate(self.P)
        self.Q = BoundPlate(self.Q, params={'mu_mean': t.zeros(())})
        self.platesizes = {'p1': 3}
        self.data = {'obs': self.data.refine_names('p1')}
        self.prob = Problem(self.P, self.Q, self.platesizes, self.data)

    def test_predictive_ll_analytic(self):
        for seed in range(10):
            t.manual_seed(seed)
            
            K = 100000
            num_samples = 100000

            sample = self.prob.sample(K, True, self.sampling_type)

            ll = sample.predictive_ll(self.extended_platesizes, True, self.extended_data, num_samples=num_samples)

            assert abs(ll['obs'] - self.true_pred_lik) < 0.01

# class TestSample_moment(unittest.TestCase):

#     def setUp(self):
#         t.manual_seed(127)

#         P = Plate(
#             ab = Group(
#                 a = Normal(0, 1),
#                 b = Normal("a", 1),
#             ),
#             c = Normal(0, lambda a: a.exp()),
#             p1 = Plate(
#                 d = Normal("a", 1),
#                 p2 = Plate(
#                     e = Normal("d", 1.),
#                 ),
#             ),
#         )

#         Q = Plate(
#             ab = Group(
#                 a = Normal("a_mean", 1),
#                 b = Normal("a", 1),
#             ),
#             c = Normal(0, lambda a: a.exp()),
#             p1 = Plate(
#                 d = Normal("d_mean", 1),
#                 p2 = Plate(
#                     e = Data()
#                 ),
#             ),
#         )

#         Q = BoundPlate(Q, params={'a_mean': t.zeros(()), 'd_mean':t.zeros(3, names=('p1',))})

#         platesizes = {'p1': 3, 'p2': 4}
#         data = {'e': t.randn(3, 4, names=('p1', 'p2'))}

#         self.prob = Problem(P, Q, platesizes, data)

#         # Get some initial samples (with K dims)
#         sampling_type = IndependentSample
#         self.K = 5
#         self.sample = self.prob.sample(self.K, True, sampling_type)

#         # Obtain K indices from posterior
#         self.num_sample = 10


#     def test_predictive_ll_analytic(self):
#         for seed in range(10):
#             t.manual_seed(seed)
            
#             moments = sample.moments({'d': [mean, var], 'c': [second_moment]})

#             assert abs(ll['obs'] - self.true_pred_lik) < 0.01
            
if __name__ == '__main__':
    unittest.main()