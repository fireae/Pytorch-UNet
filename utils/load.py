#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import random

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.jpg', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)


def get_batch_images_masks(dir_image, dir_mask, max_len=512, batch_size=4):
    image_names = os.listdir(dir_image)
    name_bases = []
    for image_name in image_names:
        name, ext = os.path.splitext(image_name)
        if ext in ['.jpg']:
            name_bases.append(image_name)
    
    batch_images = np.zeros((batch_size, 3, max_len, max_len), dtype=np.float32)
    batch_masks = np.zeros((batch_size, 1, max_len, max_len), dtype=np.float32)
    image_idx = 0
    while True:
        if image_idx >= len(name_bases):
            random.shuffle(name_bases)
            image_idx = 0

        batch_id_list = [i for i in range(batch_size)]
        if image_idx + batch_size >= len(name_bases):
            batch_id_list  = [i for i in range(len(name_bases)-image_idx-1, len(name_bases) - image_idx-1+batch_size)]
        print(batch_id_list)
        for i in batch_id_list:
            print(image_idx+i)
            if (image_idx+i) >= len(name_bases):
                random.shuffle(name_bases)
                image_idx = 0
            image_name = os.path.join(dir_image, name_bases[image_idx+i])
            mask_name= os.path.join(dir_mask, name_bases[image_idx+i])
            image = Image.open(image_name)
            mask = Image.open(mask_name)
            image = resize_and_crop(image, max_len)
            h, w = image.shape[0], image.shape[1]
            # need to transform from HWC to CHW
            imgs_switched = np.transpose(image, axes=(2, 0, 1))
   
            imgs_normalized = imgs_switched / 255.0
          
            mask = resize_and_crop(mask, max_len)
          
            print(image_name)
            batch_images[i,:, :h, :w] = np.array(imgs_normalized).astype(np.float32)
            batch_masks[i, :, :h, :w] = np.array(mask).astype(np.float32)
        
        image_idx = image_idx+batch_size
        yield batch_images, batch_masks
        batch_images = np.zeros((batch_size, 3, 736, 736), dtype=np.float32)
        batch_masks = np.zeros((batch_size, 1, 736, 736), dtype=np.float32)
        if image_idx >= len(name_bases):
            random.shuffle(name_bases)
            image_idx = 0


