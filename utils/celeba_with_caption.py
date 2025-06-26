import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, config):
        self.img_dir = config.path
        self.img_size = config.image_size

        self.valid_ext = ['png', 'jpg', 'jpeg', 'webp']
        self.img_paths = sorted([
            os.path.relpath(os.path.join(root, f), self.img_dir).replace("\\", "/")
            for root, _, files in os.walk(self.img_dir)
            for f in files
            if os.path.splitext(f)[1][1:].lower() in self.valid_ext
        ], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        self.default_caption = 'portrait of a person'
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
        with open(caption_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) < 3:
                    continue
                filename = os.path.basename(row[0])
                self.captions[filename] = row[2].strip()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.img_paths[index])
        img_basename = os.path.basename(img_path)
        caption = self.captions.get(img_basename, self.default_caption)

        rgb_img = Image.open(img_path).convert('RGB')
        img_tensor = self.T(rgb_img)

        return {
            'image': img_tensor,
            'caption': caption,
            'img_path': img_path
        }

#------------------------------------------------------------------------------
def CelebAloader(data_config, train_config, device='cuda'):
    val_split = train_config.get('validation_split', 0)  
    batch_size = train_config.get('batch_size', 32) 
    num_workers = train_config.get('num_workers', 0)
    pref_factor = train_config.get('prefetch_factor', None)

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
        pin_memory_device=device,
        persistent_workers=True,
        prefetch_factor=pref_factor
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=pref_factor
    )

    return train_loader, val_loader