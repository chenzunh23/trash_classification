# trash_classification

## Course project for Computer Vision, IIIS, Tsinghua University

This a simple platform for garbage classification. You can easily train, evaluate, and test a YOLO11 model on custom datasets.
We demonstrate the entire pipeline of segmentation tasks from data labeling all the way to reconstructing 3D objects with trained models. All related files can be found in this repository.

All our data are labeled using the TACO format.

TACO data can be downloaded from the official implementation https://github.com/pedropro/TACO

To fine-tune models on our small dataset, please load images in `data/annotations_taco.json` and split the dataset into train, val, and test datasets. It is up to you to decide how to split it!


## Requirements
ultralytics\
python>=3.8\
torch>=1.8

## Usage
### Creating datasets
You can create a dataset with the same format using files in the folder `labeling_utils`. The scripts can generate labeling interface configuration xml files for [Label Studio](https://labelstud.io/), and process the exported COCO-format JSON files into TACO format.
### Training models
To train models on our dataset, call
```
python data/YOLO_data/train_net.py --model xxx.pt
```
Other options can be found by calling argument `--help`.\
To predict with models on images, call
```
python data/YOLO_data/predict_seg.py --model xxx.pt --source your_data_path_or_list --show
```
### Visualization
We support visualization of images and annotations, the visualization code is adapted from official code of [TACO dataset](https://github.com/pedropro/TACO). When TACO dataset is downloaded, you can also visualize TACO data and train on it.
```
python vis_images.py --label_name label (visualize objects of a certain label) --dataset_path path/to/your_dataset --anns_file_path path/to/your_annotations.json --image X/y (visualize images in batched datasets, without .jpg etc) --save_path path
```
We also support visualization of reconstructed 3D objects from a particular group (works only for multi-view datasets).
```
python muti_2D_2_3D.py --group group_num --predict --model model.pt (predict 3D objects with models)
```
To run experiments using Mask-RCNN and DeepLabV3+, check out https://github.com/dbash/zerowaste, our experiments in `logs/` were performed on this codebase.
