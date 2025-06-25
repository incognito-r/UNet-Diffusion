import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pyarrow.parquet as pq
from datasets import Dataset as HFDataset

class CelebADataset(Dataset):
    def __init__(self, config):

        self.img_size = config.image_size 

        # HUggingface dataset from parquet file
        self.dataset =  HFDataset.from_parquet(config.parquet_path)

        # Load captions
        self.default_caption = 'A beautiful portrait of a person'
        self.captions = {}
        self._load_captions(config.caption_path)

        self.T = transforms.Compose([
            transforms.Resize(self.img_size),  # Resize to the specified size
            transforms.CenterCrop(self.img_size), # Ensure the image size
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def _load_captions(self, caption_file):
        if not os.path.exists(caption_file):
            raise FileNotFoundError(f"Caption file {caption_file} not found")
        with open(caption_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) < 3:
                    continue
                filename = os.path.basename(row[0])
                self.captions[filename] = row[2].strip()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]

        # Load and process image
        image = data_item['image'].convert('RGB')
        img_tensor = self.T(image)

        # Extract file name from the path (to match with captions)
        img_path = data_item['image'].filename if hasattr(data_item['image'], 'filename') else data_item.get('filepath', '')
        file_name = os.path.basename(img_path)
        
        # Get caption or default
        caption = self.captions.get(file_name, self.default_caption)

        return {
            'image': img_tensor,
            'caption': caption,
        }

#------------------------------------------------------------------------------
def CelebAloader(data_config, train_config):
    val_split = train_config.get('validation_split', 0)  
    batch_size = train_config.get('batch_size', 32) 
    num_workers = train_config.get('num_workers', 4) 

    full_dataset = CelebADataset(config=data_config)
    total_len = len(full_dataset)

    indices = list(range(total_len))
    split = int(total_len * (1 - val_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader