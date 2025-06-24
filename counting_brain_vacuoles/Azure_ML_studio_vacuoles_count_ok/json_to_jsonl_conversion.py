from collections import defaultdict

# Group annotations by image_id
grouped_annotations = defaultdict(list)

for ann in data["annotations"]:
    image = image_lookup[ann["image_id"]]
    category_name = category_lookup[ann["category_id"]]
    
    # Compute absolute bounding box coordinates
    bbox = {
        "label": category_name,
        "topX": ann["bbox"][0] * image["width"],
        "topY": ann["bbox"][1] * image["height"],
        "width": ann["bbox"][2] * image["width"],
        "height": ann["bbox"][3] * image["height"],
    }
    grouped_annotations[ann["image_id"]].append(bbox)

# Build JSONL entries with grouped annotations
grouped_jsonl_entries = []
for image_id, boxes in grouped_annotations.items():
    entry = {
        "image_url": image_lookup[image_id]["absolute_url"],
        "label": boxes
    }
    grouped_jsonl_entries.append(entry)

# Save to .jsonl format
grouped_jsonl_path = "/mnt/data/brain_vacuole_labels_grouped_azureml.jsonl"
with open(grouped_jsonl_path, "w") as f:
    for entry in grouped_jsonl_entries:
        f.write(json.dumps(entry) + "\n")

grouped_jsonl_path
