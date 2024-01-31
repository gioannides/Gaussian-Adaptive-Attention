import torch
import torch.nn as nn

from gaussian_adaptive_attention.multi_head_gaussian_adaptive_attention import MultiHeadGaussianAdaptiveAttention

class GaussianBlock(nn.Module):
    def __init__(self, norm_axes, num_heads, num_gaussians, num_layers, padding_value=None, eps=1e-8):
        super().__init__()
        if len(norm_axes) != num_layers or len(num_heads) != num_layers or len(num_gaussians) != num_layers:
            raise ValueError("Lengths of norm_axes, num_heads, and num_gaussians must match num_layers.")

        self.layers = nn.ModuleList([
            MultiHeadGaussianAdaptiveAttention(norm_axes[i], num_heads[i], num_gaussians[i], padding_value, eps)
            for i in range(num_layers)
        ])

    def forward(self, x, return_attention_details=False):
        attention_details_ = {}
        for idx, layer in enumerate(self.layers):
            if return_attention_details:
                x_, attention_details = layer(x, return_attention_details=True)
                attention_details_['layer_'+str(idx)] = attention_details
                x = x_ + x
            else:
                x = layer(x) + x

        if return_attention_details:
            return x, attention_details_
        return x
