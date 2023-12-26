from typing import Optional
import torch as t
import torch.nn as nn
from .utils import *
from .SamplingType import SamplingType, IndependentSample
from .Plate import tensordict2tree, Plate

def named2torchdim_flat2tree(flat_named:dict, all_platedims, plate):
    flat_torchdim = named2dim_dict(flat_named, all_platedims)
    return tensordict2tree(plate, flat_torchdim)

class BoundPlate(nn.Module):
    """
    Binds a Plate to inputs (e.g. film features in MovieLens) and learned parameters
    (e.g. approximate posterior parameters).

    Only makes sense at the very top layer.
        
    """
    def __init__(self, plate: Plate, inputs=None, params=None, moments=None, moment2param_init=None):
        """_summary_



        Args:
            plate (Plate): _description_
            inputs (dict, optional): _description_. Defaults to None.
            params (dict, optional): _description_. Defaults to None.
            moments (dict, optional): dict mapping from latent_name to tuple of (moment_function, moment_init, param_name), mainly used for ML update steps. Defaults to None.
            moment2param_init (dict, optional): dict mapping param_name to tuple of (moment_to_param_fn, param_init). Defaults to None.

        Raises:
            Exception: _description_
            Exception: _description_
        """
        super().__init__()
        self.plate = plate


        #Either params or moments but not both?
        assert (params is None) != (moments is None)
        assert (moment2param_init is None) == (moments is None)

        if inputs is None:
            inputs = {}
        if params is None:
            params = {}
        if moments is None:
            moments = {}
        if moment2param_init is None:
            moment2param_init = {}
        assert isinstance(inputs, dict)
        assert isinstance(params, dict)
        assert isinstance(moments, dict)
        assert isinstance(moment2param_init, dict)
        
        
        
        #Error checking: input, param names aren't reserved
        input_param_names = [*inputs.keys(), *params.keys()]
        for name in input_param_names:
            check_name(name)
        
        #Error checking: no overlap between names in inputs and params.
        inputs_params_overlap = set(inputs.keys()).intersection(params.keys())
        if 0 != len(inputs_params_overlap):
            raise Exception(f"BoundPlate has overlapping names {inputs_params_overlap} in inputs and params")

        #Error checking: no overlap between names in program and in inputs or params.
        prog_names = self.plate.all_prog_names()
        prog_input_params_overlap = set(input_param_names).intersection(prog_names)
        if 0 != len(inputs_params_overlap):
            raise Exception(f"The program in BoundPlate has names that overlap with the inputs/params.  Specifically {prog_inputs_params_overlap}.")

        for name, inp in inputs.items():
            assert isinstance(inp, t.Tensor)
            self.register_buffer(name, inp)
        for name, param in params.items():
            assert isinstance(param, t.Tensor)
            self.register_parameter(name, nn.Parameter(param))
        for name, (_, param_init) in moment2param_init.items():
            assert isinstance(param_init, t.Tensor)
            self.register_parameter(name, nn.Parameter(param_init))
        for _, (_, moment_init, param_name) in moments.items():
            ### Need to get right shape here probably...
            assert callable(moment_fn)
            self.register_buffer(param_name, moment_init)
            


    def inputs(self):
        return {k: v for (k, v) in self.named_buffers()}

    def params(self):
        return {k: v for (k, v) in self.named_parameters()}

    def inputs_params_flat_named(self):
        """
        Returns a dict mapping from str -> named tensor
        """
        return {**self.inputs(), **self.params()}

    def inputs_params(self, all_platedims:dict[str, Dim]):
        return named2torchdim_flat2tree(self.inputs_params_flat_named(), all_platedims, self.plate)

    def sample_extended(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict[str, Tensor],
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_original_platedims:list[Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:Optional[dict[str, Tensor]],
            extended_data:Optional[dict[str, Tensor]]):
        
        scope = {**scope, **self.inputs_params_flat_named()}

        return self.plate.sample_extended(
            sample,
            name,
            scope,
            inputs_params,
            original_platedims,
            extended_platedims,
            active_original_platedims,
            active_extended_platedims,
            Ndim,
            reparam,
            original_data,
            extended_data)
    
    def check_deps(self, all_platedims:dict[str, Dim]):
        self.sample(1, False, IndependentSample, all_platedims)

    def sample(self, K: int, reparam:bool, sampling_type:SamplingType, all_platedims:dict[str, Dim]):
        """
        Returns: 
            globalK_sample: sample with different K-dimension for each variable.
            logPQ: log-prob.
        """
        groupvarname2Kdim = self.plate.groupvarname2Kdim(K)

        sample = self.plate.sample(
            name=None,
            scope={},
            inputs_params=self.inputs_params(all_platedims),
            active_platedims=[],
            all_platedims=all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            sampling_type=sampling_type,
            reparam=reparam,
        )

        return sample, groupvarname2Kdim
    
    def update_params(self, K: int, reparam:bool, sampling_type:SamplingType, all_platedims:dict[str, Dim]), lr:float, num_samples:int):
        """Update params of model using self.moments and self.moment2param_init.

        Args:
            K (int): number of samples
            reparam (bool): whether to use reparameterization trick
            sampling_type (SamplingType): sampling type
            all_platedims (dict[str, Dim]): all platedims
            lr (float): learning rate
        """
        sample = self.sample(K, reparam, sampling_type, all_platedims)
        
        #Compute moments
        
        # Using importance samples
        importance_samples = sample.importance_samples(num_samples=num_samples)
        for latent_name in self.importance_samples.keys():
            for (moment_fn, moment_init, moment_name, param_name) in self.moments[latent_name]:
                moment = moment_fn(importance_samples[latent_name])

                moment_2_param = self.moment2param_init[param_name][0](moment)

                #Update moment
                self.named_buffers()[moment_name].data.add(moment, alpha=lr)
                
                #Update param
                self.params()[param_name].data.copy(moment_2_param(self.named_buffers()[moment_name]))
                
                
        # There should also be a way of doing this with the source term trick
                
                
            
        
