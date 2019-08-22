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
        # if np.sum(mask) != np.sum((mask == 1).astype(np.uint8)):
        #     print("-----")
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # print(img.shape)
        # print(mask.shape)
        # print(np.max(img))
        # print(np.min(img))
        # print(np.max(mask))
        # print(np.min(mask))
        # print("---")
    return (img, mask)

def dataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "rgb",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    save_to_dir = None, target_size = (256,256), seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle=True,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=True,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        # cv2.imwrite('image.png', img)
        # cv2.imwrite('mask.png', mask)
        # cv2.waitKey(0)
        img, mask = normalize(img,mask)
        yield (img,mask)


