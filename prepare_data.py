import numpy as np
import cv2
import os
import shutil

import config

def clear_data(data_dir):
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, 'image'))
        os.mkdir(os.path.join(data_dir, 'label'))
        
def crop_image(prefix, image, mask, save_dir):
    cnt = 0
    y = 0
    while y + config.input_size[1] <= image.shape[1]:
        x = 0
        while x + config.input_size[0] <= image.shape[0]:
            img_crop = image[x : x + config.input_size[0], y : y + config.input_size[1], :]
            mask_crop = mask[x : x + config.input_size[0], y : y + config.input_size[1], :]
            cv2.imwrite(os.path.join(save_dir, 'image', prefix + "_pos" + str(cnt) + ".png"), img_crop)
            cv2.imwrite(os.path.join(save_dir, 'label', prefix + "_pos" + str(cnt) + ".png"), mask_crop)
            x += config.strides[0]  
            cnt += 1          
        y += config.strides[1]

def process(data_dir, save_dir):
    for fname in os.listdir(os.path.join(data_dir, 'image')):
        prefix, ext = fname.split('.')
        image_path = os.path.join(data_dir, 'image', fname)
        mask_path = os.path.join(data_dir, 'label', fname)
        if ext == 'png' and os.path.isfile(mask_path):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
           
            if image.shape == mask.shape:
                crop_image(prefix, image, mask, save_dir)
            else:
                print("Shape is not match:", fname)


clear_data(config.train_dir)
clear_data(config.val_dir)

process(config.raw_train_dir, config.train_dir)
process(config.raw_val_dir, config.val_dir)