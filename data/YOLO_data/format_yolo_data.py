import json
import os
from pathlib import Path
import shutil

# This script converts the TACO dataset annotations to YOLO format.

def format_yolo_data(input_format):
    """
    Convert TACO dataset annotations to YOLO format.
    Parameters:
    input_format (str): Path to the TACO format JSON file.
    """
    BASE_DIR = Path(__file__).resolve(strict=True).parent

    if input_format == "":
        with open("../annotations_taco.json", "r") as f:
            data = json.load(f)
    else:
        with open(input_format, "r") as f:
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

def yolo_to_taco(taco_format, yolo_labels, output_json="output.json"):
    """
    Convert YOLO labels back to TACO format.

    Parameters:
    taco_format (str): Path to the TACO format JSON file.
    yolo_labels (str): Path to the YOLO labels directory.
    """
    BASE_DIR = Path(__file__).resolve(strict=True).parent

    if taco_format == "":
        with open("../annotations_taco.json", "r") as f:
            data = json.load(f)
    else:
        with open(taco_format, "r") as f:
            data = json.load(f)

    output_data = {
        "images": [],
        "annotations": [],
        "categories": data["categories"],
        "scene_categories": data["scene_categories"],
        "info": data["info"],
    }
    # Process each image in the YOLO labels directory yolo_labels
    for label_file in Path(yolo_labels).glob("*.txt"):
        file_name = label_file.stem + ".jpg"
        file_prefix = file_name.split("_")[0]
        file_name = f'Group_{file_prefix}/{file_name}'
        print(file_name)
        # Find image info in the TACO dataset
        # This doesn't assume that images are in the same directory, it just looks for the suffix
        image_info = {
            "id": len(output_data["images"]),
            "file_name": file_name,
            "width": 4096,
            "height": 3072,
        }
        output_data["images"].append(image_info)
        height = 3072
        width = 4096
        # Read the YOLO label file
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                category_id = int(parts[0])
                # Convert YOLO format to TACO format
                # YOLO format: normalized x, normalized y
                coords = [float(coord) for coord in parts[1:]]
                for i in range(len(coords)):
                    if i % 2 == 0:
                        coords[i] *= width
                    else:
                        coords[i] *= height
                segmentation = [coords]
                annotation = {
                    "id": len(output_data["annotations"]),
                    "image_id": image_info["id"],
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": [
                        min(coords[0::2]),
                        min(coords[1::2]),
                        max(coords[0::2]) - min(coords[0::2]),
                        max(coords[1::2]) - min(coords[1::2])
                    ],
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": 0,
                }
                output_data["annotations"].append(annotation)
    # Save the output data to a JSON file
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Format YOLO data")
    parser.add_argument("--input_format", type=str, default="", help="Input format json file")
    parser.add_argument("--taco_format", type=str, default="", help="TACO format json file")
    parser.add_argument("--yolo_labels", type=str, default="", help="Path to YOLO labels directory")
    args = parser.parse_args()
    if args.yolo_labels != "":
        yolo_to_taco(args.taco_format, args.yolo_labels)
    format_yolo_data(args.input_format)