import torch

# Create a tensor on CPU
cpu_tensor = torch.tensor([1, 2, 3])
print(f"Tensor on CPU: {cpu_tensor}")

# Move it to GPU (if available)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    print(f"Same tensor on GPU: {gpu_tensor}")
    print(f"Is it still the same data? {torch.equal(cpu_tensor.cpu(), gpu_tensor.cpu())}")
else:
    print("No GPU available - but your code would be much faster if there was!")
