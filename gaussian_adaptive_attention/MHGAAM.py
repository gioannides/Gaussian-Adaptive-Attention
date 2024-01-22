import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_gaussians, padding_value=None, initial_c=2, learnable_weights=True, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.eps = eps
        self.num_gaussians = num_gaussians
        self.padding_value = padding_value

        # Learnable mean offsets for each Gaussian
        self.mean_offsets = nn.Parameter(torch.zeros(num_gaussians))

        # Initialize the scale factor 'c' as a learnable parameter
        self.c = nn.Parameter(torch.full((num_gaussians,), initial_c))

        # Initialize weights
        if learnable_weights is True:
            self.weights = nn.Parameter(torch.ones(num_gaussians))
        elif isinstance(learnable_weights, torch.Tensor):
            if learnable_weights.shape[0] != num_gaussians:
                raise ValueError(f"Provided weights must have length {num_gaussians}")
            self.weights = learnable_weights
            self.register_buffer('fixed_weights', self.weights)
        else:
            raise TypeError("learnable_weights must be either True or a torch.Tensor of shape (num_gaussians,)")

    def forward(self, x):
        x = x.to(self.device)

        # Apply mask if padding value is provided
        if self.padding_value is not None:
            mask = x != self.padding_value
            x_masked = torch.where(mask, x, torch.zeros_like(x))
        else:
            x_masked = x

        # Data-derived mean and variance
        mean = x_masked.mean(dim=self.norm_axis, keepdim=True)
        var = x_masked.var(dim=self.norm_axis, keepdim=True) + self.eps

        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0) if isinstance(self.weights, nn.Parameter) else self.fixed_weights

        # Mixture of Gaussians with learned mean offsets
        mixture = 0
        for i in range(self.num_gaussians):
            adjusted_mean = mean + self.mean_offsets[i]
            y_norm = (x - adjusted_mean) / torch.sqrt(var)
            gaussian = torch.exp(-(y_norm ** 2) / (2.0 * (self.c[i] ** 2)))
            mixture += normalized_weights[i] * gaussian

        # Apply transformation
        y_transform = mixture / mixture.sum(dim=self.norm_axis, keepdim=True).clamp(min=self.eps)
        return torch.where(mask, x * y_transform, x) if self.padding_value is not None else x * y_transform



class MultiHeadGaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, learnable_weights=True, padding_value=None, eps=1e-8):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([MixtureOfGaussiansAttention(norm_axis, num_gaussians, padding_value, learnable_weights=learnable_weights, eps=eps) for _ in range(num_heads)])

    def forward(self, x):
        # Validate chunk size
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        if chunk_size == 0:
            raise ValueError("Input tensor size along norm_axis must be larger than the number of heads.")

        # Process each chunk with corresponding attention head
        return torch.cat([head(x.narrow(self.norm_axis, i * chunk_size, chunk_size)) for i, head in enumerate(self.attention_heads)], dim=self.norm_axis)
