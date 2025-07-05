from nudenet import NudeDetector
import os
from shutil import move, copy
from tqdm import tqdm

# Initialize detector
detector = NudeDetector()

# NSFW labels to flag
NSFW_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}
THRESHOLD = 0.4

# Folders
src_dir = "data/images/custom/gallery-06"
safe_dir = "data/filtered/safe"
nsfw_dir = "data/filtered/nsfw"
os.makedirs(safe_dir, exist_ok=True)
os.makedirs(nsfw_dir, exist_ok=True)

# Gather image paths recursively from all subdirectories
print(f"Scanning directory: {src_dir}")
image_paths = []
for root, dirs, files in os.walk(src_dir):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_paths.append(os.path.join(root, fname))

print(f"Found {len(image_paths)} images to process")

# Run detection in batch
print("Running NSFW detection...")
results = detector.detect_batch(image_paths=image_paths, batch_size=64)

# Process each result
safe_count = 0
nsfw_count = 0
skipped_count = 0

print("Processing results and copying files...")
for path, detections in tqdm(zip(image_paths, results), total=len(image_paths), desc="Processing images"):
    # Get relative path from src_dir to preserve folder structure
    rel_path = os.path.relpath(path, src_dir)
    
    is_nsfw = any(
        det["class"] in NSFW_CLASSES and det["score"] >= THRESHOLD
        for det in detections
    )
    
    if is_nsfw:
        # Create destination path preserving folder structure
        dest_path = os.path.join(nsfw_dir, rel_path)
        # Skip if file already exists
        if os.path.exists(dest_path):
            skipped_count += 1
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        copy(path, dest_path)
        nsfw_count += 1
    else:
        # Create destination path preserving folder structure
        dest_path = os.path.join(safe_dir, rel_path)
        # Skip if file already exists
        if os.path.exists(dest_path):
            skipped_count += 1
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        copy(path, dest_path)
        safe_count += 1

print(f"\nProcessing complete!")
print(f"Safe images: {safe_count}")
print(f"NSFW images: {nsfw_count}")
print(f"Skipped (already exist): {skipped_count}")
print(f"Total processed: {len(image_paths)}")
