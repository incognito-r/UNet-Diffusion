import pandas as pd
import io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as F
from functools import partial

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
        self.df = pd.read_parquet(config.parquet_path)

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
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 0)
    prefetch_factor = train_config.get('prefetch_factor', 2)

    dataset = ImageDataset(config=data_config)

    # loader_kwargs = {
    #     "batch_size": batch_size,
    #     "num_workers": num_workers,
    #     "pin_memory": True,
    #     "persistent_workers": num_workers > 0,
    #     "prefetch_factor": prefetch_factor if num_workers > 0 else 2,
    #     "shuffle": True
    # }

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": True
    }

    # # ✅ Use pin_memory_device if training on CUDA
    # if torch.cuda.is_available() and device.startswith("cuda"):
    #     loader_kwargs["pin_memory_device"] = device

    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader