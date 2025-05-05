import torch
import numpy as np

# From Python lists - most straightforward method
simple_tensor = torch.tensor([1, 2, 3, 4])
print(f"Simple tensor: {simple_tensor}")

# From nested lists - creates a 2D tensor
matrix_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Matrix tensor:\n{matrix_tensor}")

# Quick tensor creation functions
zeros = torch.zeros(3, 4)  # 3x4 tensor of zeros
ones = torch.ones(2, 2)    # 2x2 tensor of ones
rand = torch.rand(2, 3)    # 2x3 tensor of random values between 0 and 1
randn = torch.randn(2, 3)  # 2x3 tensor of values from standard normal distribution

print(f"\nSome tensor creation functions:")
print(f"Zeros:\n{zeros}")
print(f"Random values:\n{rand}")
