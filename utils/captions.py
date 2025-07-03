



# Merge Captions

# load captions from jsonl file
import json
import os
custom_captions_path = 'data/custom_captions.jsonl'
celebA_captions_path = 'data/celebA_captions.jsonl'

custom_captions = {}
if os.path.exists(custom_captions_path):
    with open(custom_captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            custom_captions[entry["file_name"]] = entry["text"]

celebA_captions = {}
if os.path.exists(celebA_captions_path):
    with open(celebA_captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            celebA_captions[entry["file_name"]] = entry["text"]

# merge clebA captions with custom captions
merged_captions = {**celebA_captions, **custom_captions}

# save merged captions to jsonl file
with open('data/captions.jsonl', 'w', encoding='utf-8') as f:
    for file, caption in merged_captions.items():
        entry = {"file_name": file, "text": caption}
        f.write(json.dumps(entry) + '\n')

print(f"CelebA Captions: {len(celebA_captions)} | Custom Captions: {len(custom_captions)} | Total Captions: {len(merged_captions)}")
print(f"Caption file Saved!")