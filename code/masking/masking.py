import math 
import json
import warnings

import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

# names for plt layers
_bert_names = {
    "K": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.key", 
    "Q": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.query", 
    "V": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.value", 
    "AO": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.output.dense", 
    "I": lambda ptl, l: f"{ptl}.encoder.layer.{l}.intermediate.dense", 
    "O": lambda ptl, l: f"{ptl}.encoder.layer.{l}.output.dense", 
    "P": lambda ptl, l: f"{ptl}.pooler.dense", 
}

def chain_module_names(which_ptl, layer_indices, abbres):
    names = set()
    for abbre in abbres:
        for l in layer_indices:
            names.add(_bert_namesnames[abbre](which_ptl, l))


class MaskedLinearX(nn.Module):
    def __init__(self, scheme_idx, weight, bias, mask_biases, **kwargs):
        super(MaskedLinearX, self).__init__()
        init_scale = kwargs["init_scale"]
        init_sparsity = kwargs["init_sparsity"]
        self.name = kwargs["name"]
        self.threshold = kwargs["threshold"]
        self.threshold_fn = _Binarizer1
        self.mask_biases = mask_biases
        

        self.weight = weight
        self.bias = bias
        self.structured_mask_expanding = None
        self._controlled_init = kwargs["controlled_init"]


        #init mask
        structured_masking_info = kwargs["structured_masking_info"]
        structured_masking = structured_masking_info["structured_masking"]
        structured_masking_types = structured_masking_info["structured_masking_types"]
        self.force_masking = structured_masking_info["force_masking"]
        self.is_structured_masking = True
        

        # adjust random initiatilization scale of the mask to satisfy initial sparsity
        init_scale = self.get_init_scales(init_sparsity, init_scales)

        # structured pruning
        self.structured_masked = False
        if structured_masking == "layers":
            _template = torch.FloatTensor([1]).uniform_(*init_scales)
            self.weight_mask = nn.Parameters(_template.clone())
        elif structured_masking == "heads":
            assert "self" in self.name

            conf = structured_masking_info["ptl_config"]
            num_attention_heads = conf.num_attention_heads
            attention_head_size = int(conf.hidden_size / num_attention_heads)

            # init masks
            _template = torch.FloatTensor([1]  * num_attention_heads).uniform_(
                *init_scales
            )

            self.structured_mask_expanding = nn.Parameters(
                torch.ones(num_attention_heads, attention_head_size), 
                requires_grad=False
            )

            self.weight_mask = nn.Parameter(_template.clone())
    
    def controlled_init(self, weight, init_sparsity, threshold, controlled_init_type):
        # i think i'm gonna die in this house

        #get threshold by magnitude
        _weight_size = weight.nelement()
        _num_zero_element = int(_weight_size * init_sparsity)

        def _uniform():
            """randomly sample indices from the tensor to assign values for the mask - init mask from uniform distribution"""
            _weight = torch.zeros_like(weight.view(-1))
            indices = np.arange(_weight_size)
            sampled_indices = np.random.choice(indices, size=_num_zero_element)
            _bool_masks = torch.ones_like(_weight)
            _bool_masks[sampled_indices] = 0
            _bool_masks = _bool_masks.bool()

            _weight[_bool_masks] = 2.0 * threshold
            _weight[~_bool_masks] = 0.0 * threshold
            return _weight.view(*weight.size())
        
        weight_mask = _uniform()


    def get_init_scales(self, init_sparsity, init_scale):
        s = (init_scale + self.threshold) / init_sparsity - init_scale
        return (-init_scale, s)

    
    """now we get to mask bert!!"""

    def reshape_mask_for_sp(mask, structured_mask_expanding, name="weight"):
        if structured_mask_expanding is not None: 
            _mask = (mask.unsqueeze(1) * structured_mask_expanding).view(-1)
            if name == "weight":
                mask = _mask.unsqueeze(1)
            elif name == "bias":
                mask = _mask.unsqueeze(0).unsqueeze(0)
        return mask

# our no mask for the baseline - regular model
class MaskedLinear0(nn.Module):
    def __init__(self, weight, bias, **kwargs):
        super(MaskedLinear0, self).__init__():
        self.weight = weight
        self.bias = bias
    
    def forward(self.x):
        return F.linear(x, self.weight, self.bias)


# our masking model!! use partial derivative of Loss wrt binarized version of mask to approx Loss wrt mask

def binarizer_fn1(inputs, threshold):
    outputs = inputs.clone()
    outputs[inputs.le(threshold)] = 0.0
    outputs[inputs.gt(threshold)] = 1.0
    return

class _Binarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        return binarized_fn1(inputs, threshold)
    
    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None, None)

class MaskedLinear1(nn.Module):
    def __init__(self, weight, bias, mask_biases, **kwargs):
        super(MaskedLinear1, self).__init__(
            "MaskedLinear1", weight, bias, mask_biases, **kwargs
        )
    
    def get_masks(self):
        M_w = self.threshold_fn(self.weight_mask, self.threshold)
        M_w = reshape_mask_for_sp(M_w, self.structured_mask_expanding, name="weight")

        return M_w
    
    def forward(self.x):
        M_w = self.get_masks()
        return F.linear(x, self.weight * M_w, self.bias)