import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

# === Configuration ===
img_dir = 'data/images/custom'
captions_path = 'data/captions.jsonl'
output_dir = 'data/parquet/'
shard_prefix = 'dataset_'  # Prefix for shard filenames
shard_size = 10_000
resize_to = (512, 512)  # Set your target resolution here
image_quality = 95  # JPEG quality
fill_last_shard = True  # If True, fill the existing last shard before creating new ones

# Smart resize function
def smart_resize(img, target_size):
    """Smart resize that uses LANCZOS for downscaling and BICUBIC for upscaling"""
    w, h = img.size
    if w > target_size[0] and h > target_size[1]:
        # Downscaling - use LANCZOS for better quality
        return img.resize(target_size, resample=Image.Resampling.LANCZOS)
    else:
        # Upscaling - use BICUBIC for better quality
        return img.resize(target_size, resample=Image.Resampling.BICUBIC)

os.makedirs(output_dir, exist_ok=True)
print("📖 Loading captions...")

# === Load Captions ===
captions = {}
with open(captions_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        captions[entry["file_name"]] = entry["text"]
print(f"✅ Loaded {len(captions)} captions.")

# === Collect existing image_ids to skip duplicates ===
print("🔍 Scanning existing shards for duplicate image_ids...")
existing = sorted(f for f in os.listdir(output_dir) if f.endswith('.parquet') and f.startswith(shard_prefix))
existing_image_ids = set()
for fname in tqdm(existing, desc="Reading shards"):
    df = pd.read_parquet(os.path.join(output_dir, fname), columns=['image_id'])
    existing_image_ids.update(df['image_id'].tolist())
print(f"✅ Found {len(existing_image_ids)} existing image_ids to skip.")

# === Gather and Resize Images ===
data = []
skipped = 0
processed = 0
valid_ext = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff'}
print("🖼️  Processing and resizing images...")

for root, _, files in os.walk(img_dir):
    for fname in tqdm(files):
        ext = os.path.splitext(fname)[1][1:].lower()
        if ext not in valid_ext:
            continue

        image_id = os.path.splitext(fname)[0]
        if image_id in existing_image_ids:
            skipped += 1
            continue  # Skip duplicate

        caption = captions.get(fname, "")
        img_path = os.path.join(root, fname)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = smart_resize(img, resize_to)

                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=image_quality)
                img_bytes = buffer.getvalue()

                data.append({'image_id': image_id, 'image': img_bytes, 'text': caption})
                processed += 1
        except Exception as e:
            print(f"[Warning] Skipped {img_path}: {e}")

print(f"✅ Finished image processing: {processed} new images ready. {skipped} duplicates skipped.")

# === Determine last shard state ===
last_index = -1
last_count = 0
if existing:
    last_index = max(int(f.split('_')[-1].split('.')[0]) for f in existing)
    last_path = os.path.join(output_dir, f"{shard_prefix}{last_index:03d}.parquet")
    last_df = pd.read_parquet(last_path)
    last_count = len(last_df)

# === Optionally fill the last shard ===
start_index = last_index + 1
if fill_last_shard and last_count > 0 and last_count < shard_size:
    to_fill = min(shard_size - last_count, len(data))
    if to_fill > 0:
        fill_df = pd.DataFrame(data[:to_fill])
        updated_last = pd.concat([last_df, fill_df], ignore_index=True)
        updated_last.to_parquet(last_path, index=False)
        print(f"📦 Filled shard {last_index:03d} to {len(updated_last)} images.")
        data = data[to_fill:]
        if len(updated_last) == shard_size:
            start_index = last_index + 1
        else:
            start_index = last_index

# === Split remaining data into new shards ===
total_new = len(data)
num_new_shards = (total_new + shard_size - 1) // shard_size
print(f"💾 Saving {total_new} remaining images into {num_new_shards} new shards...")

for i in tqdm(range(num_new_shards), desc="Saving new shards"):
    s_idx = start_index + i
    start = i * shard_size
    end = min(start + shard_size, total_new)
    shard_data = data[start:end]
    df = pd.DataFrame(shard_data)
    shard_path = os.path.join(output_dir, f"{shard_prefix}{s_idx:03d}.parquet")
    df.to_parquet(shard_path, index=False)
    print(f"✅ Saved shard {s_idx:03d} with {len(df)} images to {shard_path}")

print("🎉 All done! Updated shards saved in:", output_dir)
