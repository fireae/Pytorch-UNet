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

    imgs = to_cropped_imgs(ids, dir_img, ".jpg", scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, "_mask.jpg", scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + ".jpg")
    mask = Image.open(dir_mask + id + "_mask.gif")
    return np.array(im), np.array(mask)


def get_batch_images_masks(dir_image, dir_mask, max_len=512, batch_size=4):
    image_names = os.listdir(dir_image)
    name_bases = []
    for image_name in image_names:
        name, ext = os.path.splitext(image_name)
        if ext in [".jpg"]:
            name_bases.append(name)

    batch_images = np.zeros((batch_size, 3, max_len, max_len), dtype=np.float32)
    batch_masks = np.zeros((batch_size, 1, max_len, max_len), dtype=np.float32)
    image_index = np.arange(0, len(name_bases))
    while True:
        np.random.shuffle(image_index)
        batch_images = []
        batch_masks = []
        for i in image_index:
            image_name = os.path.join(dir_image, name_bases[i] + ".jpg")
            mask_name = os.path.join(dir_mask, name_bases[i] + "_mask.jpg")
            image = Image.open(image_name)
            mask = Image.open(mask_name)
            image = resize_and_crop(image, max_len)
            h, w = image.shape[0], image.shape[1]
            # need to transform from HWC to CHW
            imgs_switched = np.transpose(image, axes=(2, 0, 1))

            imgs_normalized = imgs_switched / 255.0

            mask = resize_and_crop(mask, max_len)
            mask = mask / 255.0
            batch_images.append(imgs_normalized)
            batch_masks.append(mask)

            if len(batch_images) == batch_size:
                print(len(batch_images))
                print(len(batch_masks))
                np_batch_images = np.ones(
                    (len(batch_images), 3, max_len, max_len), dtype=np.float32
                )
                np_batch_masks = np.zeros(
                    (len(batch_images), 1, max_len, max_len), dtype=np.float32
                )
                for i in range(len(batch_images)):
                    h, w = batch_images[i].shape[1], batch_images[i].shape[2]
                    np_batch_images[i, :, :h, :w] = batch_images[i]
                    np_batch_masks[i, :, :h, :w] = batch_masks[i]
                yield np_batch_images, np_batch_masks
                batch_images = []
                batch_masks = []

