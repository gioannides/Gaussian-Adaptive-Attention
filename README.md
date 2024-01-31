# Multi-Head Gaussian Adaptive Attention

GaussianAdaptiveAttention is a PyTorch library providing modules for applying Gaussian adaptive attention mechanisms that can approximate any Probability Distribution to derive attention weights. This library offers a novel approach to enhance neural network models with adaptive attention capabilities, inspired by Gaussian distribution principles.

## Features

- **Customizable Gaussian Attention**: Tailor the attention mechanism to suit various neural network architectures.
- **Multiple Attention Heads**: Support for multiple attention heads for complex tasks.
- **PyTorch Integration**: Seamlessly integrates with existing PyTorch models.

## Installation

Install GaussianAdaptiveAttention easily using pip:

```bash
pip install gaussian-adaptive-attention==0.1.3
```

OR

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
from gaussian_adaptive_attention import GaussianAdaptiveAttention, MultiHeadGaussianAdaptiveAttention, GaussianBlock
```

# Example Usage
This example demonstrates the use of the `GaussianBlock` class, which encapsulates multiple layers of Gaussian Adaptive Attention.

```python
import torch
import torch.nn as nn
from gaussian_adaptive_attention.gaussian_block import GaussianBlock

norm_axes = [1, 1, 1]  # Axes for each layer in the GaussianBlock.
num_heads = [4, 4, 4]  # Number of attention heads for each layer.
num_gaussians = [5, 5, 5]  # Number of Gaussians per head for each layer.
num_layers = 3  # Total number of layers in the GaussianBlock.
padding_value = None  # Padding value for sequences in the input tensor.
eps = 1e-8  # Small epsilon value for numerical stability.

# Initialize the GaussianBlock
attention_block = GaussianBlock(norm_axes, num_heads, num_gaussians, num_layers, padding_value, eps)

# Example neural network with GaussianBlock
class ExampleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExampleNetwork, self).__init__()
        # Initialize GaussianBlock for attention mechanism
        self.attention_block = attention_block
        # Initialize a linear layer
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Apply GaussianBlock for attention
        x = self.attention_block(x)
        # Apply linear layer
        x = self.linear(x)
        return x

# Example usage
input_dim = 128
output_dim = 128
model = ExampleNetwork(input_dim, output_dim)
input_tensor = torch.rand(10, input_dim)  # Example input tensor
output = model(input_tensor)
```

## Paper

If you use the Gaussian Adaptive Attention, please cite our paper (https://arxiv.org/abs/2401.11143). The source code for the experiments in the paper is coming soon!

```
@misc{ioannides2024gaussian,
      title={Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities}, 
      author={Georgios Ioannides and Aman Chadha and Aaron Elkins},
      year={2024},
      eprint={2401.11143},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
        
## Contributing

Contributions to the GaussianAdaptiveAttention library are welcome!

-    **Report Issues**: Submit an issue on GitHub if you find bugs or potential improvements.
-    **Submit Pull Requests**: Feel free to fork the repository and submit pull requests with your enhancements.

## License

This project is licensed under Apache-2.0 - see the LICENSE file in the repository for details.
