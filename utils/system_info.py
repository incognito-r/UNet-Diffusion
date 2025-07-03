import os
import psutil
import torch

# CPU info
cpu_count = os.cpu_count()
print(f"CPU cores: {cpu_count}")

# RAM info
ram_bytes = psutil.virtual_memory().total
ram_gb = ram_bytes / (1024 ** 3)
print(f"System RAM: {ram_gb:.2f} GB")

# GPU info (if available)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        vram_bytes = torch.cuda.get_device_properties(i).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        print(f"GPU {i}: {gpu_name} - VRAM: {vram_gb:.2f} GB")
else:
    print("No CUDA-compatible GPU found.")