import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers


class FactorizedLayerNorm(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

        Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        do_projection: Weather to project the input.
        do_scale: Weather to scale te input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`[* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
            \times \ldots \times \text{normalized\_shape}[-1]]`.
        gain: the learnable weights of the affine transformation of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the affine transformation of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    """
    def __init__(self, normalized_shape, elementwise_affine=True, device=None, dtype=None, do_scale=True, do_projection=True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FactorizedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.prod_normalized_shape = torch.prod(torch.tensor(self.normalized_shape)).long()
        self.prod_normalized_shape_sqrt = self.prod_normalized_shape ** 0.5
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('gain', None)
            self.register_parameter('bias', None)
        self.do_projection = do_projection
        self.do_scale = do_scale

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.gain)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        batch_dimensions = input.shape[:-len(self.normalized_shape)]
        # (batch_size, prod_normalized_shape)
        reshaped_input = input.reshape((-1, self.prod_normalized_shape))
        if self.do_projection:
            # (1, prod_normalized_shape)
            ones = torch.ones((1, self.prod_normalized_shape), device=input.device)
            # (1)
            sqr_size = torch.linalg.norm(ones, dim=1) ** 2
            # (1, prod_normalized_shape))
            scaled_ones = ones / sqr_size
            # (batch_size, 1)
            dot_prod = F.linear(reshaped_input, ones)
            # (batch_size, prod_normalized_shape)
            sub = dot_prod @ scaled_ones
            reshaped_input = reshaped_input - sub
        
        if self.do_scale:
            reshaped_input = F.normalize(reshaped_input, dim=1) * self.prod_normalized_shape_sqrt
        
        reshaped_input = reshaped_input.reshape(*batch_dimensions, *self.normalized_shape)
    
        if self.elementwise_affine:
            reshaped_input *= self.gain
            reshaped_input += self.bias
        return reshaped_input
