import os
import json

# root directory
LOCAL_FILES_ROOT = "./GroupedPhotos"

num_groups = 45
num_images_per_group = 4

image_files = []

for i in range(num_groups):
    for j in range(num_images_per_group):
        image_files.append(f"data/local-files?d=Group_{i}/{i}_{j}.jpg")

tasks = []
for img_path in image_files:
    task = {
        "data": {
            "image": img_path,
            "metadata": {
                "width": 4096,
                "height": 3072
            }
        },
        "annotations": []  
    }
    tasks.append(task)

with open(os.path.join(LOCAL_FILES_ROOT, "all_images_tasks.json"), "w") as f:
    json.dump(tasks, f, indent=2)