import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab

import argparse

def vis_images(label_name):
    """
        Visualize images with annotations
        Adapted from official implementation of TACO dataset
    """
    dataset_path = './GroupedPhotos'
    anns_file_path = dataset_path + '/' + 'annotations_taco.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    n_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = n_super_cats
            super_cat_last_name = super_cat_name
            n_super_cats += 1

    # User settings
    n_img_2_display = 10
    category_name = label_name
    pylab.rcParams['figure.figsize'] = (14,14)

    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)

    # Get image ids
    imgIds = []
    catIds = coco.getCatIds(catNms=[category_name])
    if catIds:
        # Get all images containing an instance of the chosen category
        imgIds = coco.getImgIds(catIds=catIds)
    else:
        # Get all images containing an instance of the chosen super category
        catIds = coco.getCatIds(supNms=[category_name])
        for catId in catIds:
            imgIds += (coco.getImgIds(catIds=catId))
        imgIds = list(set(imgIds))

    n_images_found = len(imgIds) 
    print('Number of images found: ',n_images_found)

    # Select N random images
    random.shuffle(imgIds)
    imgs = coco.loadImgs(imgIds[0:min(n_img_2_display,n_images_found)])

    for img in imgs:
        image_path = dataset_path + '/' + img['file_name']
        # Load image
        I = Image.open(image_path)
        
        # Load and process image metadata
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)
        
        # Show image
        fig,ax = plt.subplots(1)
        plt.axis('off')
        plt.imshow(I)

        # Load mask ids
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns_sel = coco.loadAnns(annIds)
        
        # Show annotations
        for ann in anns_sel:
            color = colorsys.hsv_to_rgb(np.random.random(),1,1)
            for seg in ann['segmentation']:
                poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
                p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
                ax.add_collection(p)
                p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
                ax.add_collection(p)
            [x, y, w, h] = ann['bbox']
            rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
                            facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)
        plt.show()
        plt.savefig('vis_images/' + img['file_name'].split('/')[-1].split('.')[0] + '.png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize images with annotations')
    parser.add_argument('--label_name', type=str, default='Bottle',
                        help='Name of the label to visualize')
    args = parser.parse_args()
    vis_images(args.label_name)
