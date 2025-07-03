import os
import json
import pandas as pd

img_dir = 'data/images/CelebA/high_res'
captions_path = 'data/captions.jsonl'
parquet_path = 'data/celebA_dataset_high.parquet'

print("Processing...")
# Load captions
captions = {}
with open(captions_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        captions[entry["file_name"]] = entry["text"]

# Gather image bytes, captions, and image IDs
data = []
valid_ext = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff', 'gif'}
for root, _, files in os.walk(img_dir):
    for fname in files:
        ext = os.path.splitext(fname)[1][1:].lower()
        if ext in valid_ext:
            caption = captions.get(fname, "")
            img_path = os.path.join(root, fname)
            with open(img_path, "rb") as img_f:
                img_bytes = img_f.read()
            image_id = os.path.splitext(fname)[0]  # filename without extension
            data.append({'image_id': image_id, 'image': img_bytes, 'text': caption})

# Save to Parquet
df = pd.DataFrame(data)
df.to_parquet(parquet_path, index=False)
print(f"Saved {len(df)} images, captions, and image_ids to {parquet_path}")