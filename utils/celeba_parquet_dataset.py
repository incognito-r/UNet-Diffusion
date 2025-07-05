import pandas as pd
import io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as F
from functools import partial
import pyarrow.parquet as pq
import pyarrow as pa
import os
import glob

class SmartResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if w > self.size[0] and h > self.size[1]:
            return F.resize(img, self.size, interpolation=transforms.InterpolationMode.LANCZOS)
        else:
            return F.resize(img, self.size, interpolation=transforms.InterpolationMode.BICUBIC)

class ImageDataset(Dataset):
    def __init__(self, config):
        self.default_caption = 'portrait of a person'
        
        # Fast loading of multiple parquet files using PyArrow
        if os.path.isdir(config.parquet_path):
            # Load all parquet files in the directory
            parquet_pattern = os.path.join(config.parquet_path, "*.parquet")
            parquet_files = sorted(glob.glob(parquet_pattern))
            
            if not parquet_files:
                raise ValueError(f"No parquet files found in {config.parquet_path}")
            
            print(f"Loading {len(parquet_files)} parquet files using PyArrow...")
            
            # Use PyArrow to read all files in parallel
            dataset = pq.ParquetDataset(parquet_files)
            table = dataset.read()
            self.df = table.to_pandas()
            
            print(f"✅ Loaded {len(self.df)} total samples from {len(parquet_files)} files")
        else:
            # Single file
            print(f"Loading single parquet file: {config.parquet_path}")
            self.df = pd.read_parquet(config.parquet_path, engine='pyarrow')
            print(f"✅ Loaded {len(self.df)} samples")

        self.T = transforms.Compose([
            SmartResize((config.image_size, config.image_size)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']
        caption = row.get('text', self.default_caption)
        image_id = row.get('image_id', None)
        
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = self.T(image)

        return {
            'image': img_tensor,
            'text': caption,
            'image_id': image_id
        }

#=============================================================================

def DatasetLoader(data_config, train_config, device='cuda'):
    val_split = train_config.get('validation_split', 0)  
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 0)
    prefetch_factor = train_config.get('prefetch_factor', 2)

    full_dataset = ImageDataset(config=data_config)
    total_len = len(full_dataset)

    indices = list(range(total_len))
    split = int(total_len * (1 - val_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else 2,
        "shuffle": True
    }

    # ✅ Use pin_memory_device if training on CUDA
    if torch.cuda.is_available() and device.startswith("cuda"):
        loader_kwargs["pin_memory_device"] = device

    train_loader = DataLoader(train_subset, **loader_kwargs)


    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader