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

def vis_image(image, save_path, dataset_path = './GroupedPhotos', anns_file_path = './GroupedPhotos/annotations_taco.json', vis=False):
    """
        Visualize images with annotations
        
        Adapted from official implementation of TACO dataset https://github.com/pedropro/TACO

        Args:
            image (str): image file name of a specific format, i.e., X/Y.
            anns_file_path (str): Path to the annotations file.
            save_path (str): Path to save the visualized images.
    """
    
    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)

    # Find image id
    img_id = -1
    for img in imgs:
        if image == img['file_name']:
            img_id = img['id']
            break
            
    # Show image and corresponding annotations
    if img_id == -1:
        print('Incorrect file name')
    else:

        # Load image
        image_prefix = image.split('_')[0]
        I = Image.open(dataset_path + '/' + image)

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
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
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
        if vis:
            plt.show()
        if save_path != '':
            plt.savefig(save_path +'/'+ image.split('/')[-1].split('.')[0] + '_out.png', dpi=300)



def vis_images_with_categories(label_name, save_path, dataset_path = './GroupedPhotos', anns_file_path = './GroupedPhotos/annotations_taco.json', vis=False):
    """
        Visualize images with annotations
        Adapted from official implementation of TACO dataset https://github.com/pedropro/TACO
    """
    
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
    n_img_2_display = 5
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
        if vis:
            plt.show()
        plt.savefig(save_path + '/' + img['file_name'].split('/')[-1].split('.')[0] + '_seg.jpg', dpi=300)

def vis_3d_images():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize images with annotations')
    parser.add_argument('--label_name', type=str, default='Bottle',
                        help='Name of the label to visualize')
    parser.add_argument('--dataset_path', type=str, default='./GroupedPhotos',
                        help='Path to the dataset')
    parser.add_argument('--anns_file_path', type=str, default='./GroupedPhotos/annotations_taco.json',
                        help='Path to the annotations file')
    parser.add_argument('--save_path', type=str, default='',
                        help='Path to save the visualized images')
    parser.add_argument('--image', type=str, default='Group_11/11_1.jpg',
                        help='Image file name of a specific format, i.e., X_Y')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize images with annotations')
    args = parser.parse_args()
    vis_image(args.image, args.save_path, args.dataset_path, args.anns_file_path, args.vis)
    vis_images_with_categories(args.label_name, args.save_path, args.dataset_path, args.anns_file_path, args.vis)
