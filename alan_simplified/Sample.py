from torch.autograd import grad
from functorch.dim import Dim

from .SamplingType import SamplingType
from .Split import Split
from .Plate import Plate, tensordict2tree, flatten_tree, empty_tree
from .utils import *
from .logpq import logPQ_plate

class Sample():
    def __init__(
            self,
            problem,
            sample: dict,
            groupvarname2Kdim: dict[str, Dim],
            sampling_type: SamplingType,
            split:Split
        ):
        self.problem = problem
        self.sample = sample
        self.groupvarname2Kdim = groupvarname2Kdim
        self.sampling_type = sampling_type
        self.split = split

    @property 
    def P(self):
        return self.problem.P

    @property 
    def Q(self):
        return self.problem.Q

    @property 
    def all_platedims(self):
        return self.problem.all_platedims

    def elbo(self, extra_log_factors=None):

        if extra_log_factors is None:
            extra_log_factors = empty_tree(self.P)
        assert isinstance(extra_log_factors, dict)
        #extra_log_factors = named2dim_dict(extra_log_factors, self.all_platedims)
        #extra_log_factors = tensordict2tree(self.P, extra_log_factors)

        lp = logPQ_plate(
            name=None,
            P=self.P, 
            Q=self.Q, 
            sample=self.sample,
            inputs_params_P=self.P.inputs_params(self.all_platedims),
            inputs_params_Q=self.Q.inputs_params(self.all_platedims),
            data=self.problem.data,
            extra_log_factors=extra_log_factors,
            scope_P={}, 
            scope_Q={}, 
            active_platedims=[],
            all_platedims=self.all_platedims,
            groupvarname2Kdim=self.groupvarname2Kdim,
            sampling_type=self.sampling_type,
            split=self.split)

        return lp

    def marginals(self):
        #This is an ordered dict.
        samples = flatten_tree(self.sample) 

        #List of named Js to go into torch.autograd.grad
        Js_named_list = []
        #Flat dict of torchdim tensors to go into elbo as extra_log_factors
        Js_torchdim_dict = {}
        #dimension names
        dimnamess = []
        for (varname, sample) in samples.items():
            dims = generic_dims(sample)
            shape = [dim.size for dim in dims]
            dimnames = [str(dim) for dim in dims]
            dimnamess.append(dimnames)
            J_named = t.zeros(*shape, device=sample.device, requires_grad=True, names=dimnames)
            Js_named_list.append(J_named)
            J_torchdim = J_named.rename(None)[dims]
            #Marginals need different names from variables.
            #This is really a problem in how we're representing trees...
            Js_torchdim_dict[f"{varname}_marginal"] = J_torchdim
        Js_torchdim_tree = tensordict2tree(self.P, Js_torchdim_dict)

        #Compute loss
        L = self.elbo(extra_log_factors=Js_torchdim_tree)
        #marginals as a list
        marginals_list = grad(L, Js_named_list)

        #marginals as a flat dict
        marginals_dict = {}
        for varname, marginal, dimnames in zip(samples.keys(), marginals_list, dimnamess):
            marginals_dict[varname] = marginal.refine_names(*dimnames)

        return marginals_dict
    


