from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import cv2
import config



def normalize(img, mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)

def dataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, 
                image_save_prefix  = "image", mask_save_prefix  = "mask",
                save_to_dir = None, target_size = (256,256), seed = None, shuffle=True):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    if seed is None:
        seed = np.random.choice(range(9999))
    flow_args = dict(
        target_size = target_size, 
        batch_size = batch_size, 
        seed = seed,
        shuffle = shuffle,
        save_to_dir = save_to_dir)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        save_prefix = image_save_prefix,
        **flow_args)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        save_prefix = mask_save_prefix,
        **flow_args)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = normalize(img,mask)
        yield (img,mask)


