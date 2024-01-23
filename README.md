# Multi-Head Gaussian Adaptive Attention

GaussianAdaptiveAttention is a PyTorch library providing modules for applying Gaussian adaptive attention mechanisms that can approximate any Probability Distribution to derive attention weights. This library offers a novel approach to enhance neural network models with adaptive attention capabilities, inspired by Gaussian distribution principles.

## Features

- **Customizable Gaussian Attention**: Tailor the attention mechanism to suit various neural network architectures.
- **Multiple Attention Heads**: Support for multiple attention heads for complex tasks.
- **PyTorch Integration**: Seamlessly integrates with existing PyTorch models.

## Installation

Install GaussianAdaptiveAttention easily using pip:

```bash
pip3 install git+https://github.com/gioannides/GaussianAdaptiveAttention.git
```

## Requirements

- Python 3.x
- PyTorch (latest version recommended)

## Usage

Import the GaussianAdaptiveAttention and MultiHeadGaussianAdaptiveAttention modules and integrate them into your PyTorch models:

```python3
import torch
from gaussian_adaptive_attention import GaussianAdaptiveAttention, MultiHeadGaussianAdaptiveAttention
```

# Example usage
```
norm_axis = 1 # Axis for learning Distributions on
num_gaussians = 5 # Example number of Gaussians per head
learnable_weights = True # Adjustable weights for the Gaussian Mixture, if set to False then the weights have to be provided as torch.full((num_gaussians,)
initial_c = 2 # The learnable scaled variance initial value
padding_value = None # The value that the sequence has been padded with to be ignored during statistical parameter estimation. None implies no padding.
eps = 1e-8 # The value to stabilize training (small epsilon value)

attention_module = MultiHeadGaussianAdaptiveAttention(norm_axis, num_heads, num_gaussians, \
                                            initial_c, learnable_weights, padding_value, eps)
```

## Example

Here's a simple example demonstrating how to apply the GaussianAdaptiveAttention module in a neural network layer.

```
import torch
import torch.nn as nn
from gaussian_adaptive_attention import GaussianAdaptiveAttention, MultiHeadGaussianAdaptiveAttention

# Example neural network layer with Gaussian Adaptive Attention
class ExampleNetwork(nn.Module):
    def __init__(self, ...):
        super(ExampleNetwork, self).__init__()
        # Initialize network layers here
        self.attention = MultiHeadGaussianAdaptiveAttention(...)

    def forward(self, x):
        # Apply layers and attention
        x = self.attention(x)
        return x
```
        
## Contributing

Contributions to the GaussianAdaptiveAttention library are welcome!

-    **Report Issues**: Submit an issue on GitHub if you find bugs or potential improvements.
-    **Submit Pull Requests**: Feel free to fork the repository and submit pull requests with your enhancements.

## License

This project is licensed under Apache-2.0 - see the LICENSE file in the repository for details.
