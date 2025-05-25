import json
import os
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open("../annotations_taco.json", "r") as f:
    data = json.load(f)

Path("./labels/train").mkdir(parents=True, exist_ok=True)
Path("./labels/val").mkdir(parents=True, exist_ok=True)
Path("./labels/test").mkdir(parents=True, exist_ok=True)

categories = {cat["id"]: i for i, cat in enumerate(data["categories"])}

def get_existing_files(split):
    return set(f.name for f in (BASE_DIR / "images" / split).glob("*"))

existing_files = {
    "train": get_existing_files("train"),
    "val": get_existing_files("val"),
    "test": get_existing_files("test")
}

for img in data["images"]:
    file_name = Path(img["file_name"]).name
    width = img["width"]
    height = img["height"]
    
    if file_name in existing_files["train"]:
        split = "train"
    elif file_name in existing_files["val"]:
        split = "val"
    elif file_name in existing_files["test"]:
        split = "test"
    else:
        print(f"Skip unclassified files: {file_name}")
        continue
    
    label_path = BASE_DIR / "labels" / split / f"{Path(file_name).stem}.txt"
    
    annotations = [anno for anno in data["annotations"] if anno["image_id"] == img["id"]]
    
    with open(label_path, "w") as f:
        for anno in annotations:
            if anno["iscrowd"] == 1:
                continue

            category_id = categories[anno["category_id"]]
            
            # COCO style segmentation
            segments = anno["segmentation"]
            
            for segment in segments:
                normalized_segment = [
                    round(coord / width if i%2 == 0 else coord / height, 6)
                    for i, coord in enumerate(segment)
                ]
                
                line = f"{category_id} " + " ".join(map(str, normalized_segment)) + "\n"
                f.write(line)