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
    cnt_water = 0
    cnt_land = 0
    y = 0
    while y + config.input_size[1] <= image.shape[1]:
        x = 0
        while x + config.input_size[0] <= image.shape[0]:
            img_crop = image[x : x + config.input_size[0], y : y + config.input_size[1], :]
            mask_crop = mask[x : x + config.input_size[0], y : y + config.input_size[1], :]
            x += config.strides[0]  
            if np.sum((mask_crop==255).astype(np.uint8)) / 3 >= config.min_num_pixel:
                cnt_water += 1     
                cv2.imwrite(os.path.join(save_dir, 'image', prefix + "_water_" + str(cnt_water) + ".png"), img_crop)
                cv2.imwrite(os.path.join(save_dir, 'label', prefix + "_water_" + str(cnt_water) + ".png"), mask_crop)
        #     elif cnt_land < cnt_water and np.random.choice(100) < 25:
        #         cnt_land += 1
        #         cv2.imwrite(os.path.join(save_dir, 'image', prefix + "_land_" + str(cnt_land) + ".png"), img_crop)
        #         cv2.imwrite(os.path.join(save_dir, 'label', prefix + "_land_" + str(cnt_land) + ".png"), mask_crop)

        y += config.strides[1]

def process(data_dir, save_dir, scale_range=[]):
    for fname in os.listdir(os.path.join(data_dir, 'image')):
        prefix, ext = fname.split('.')
        image_path = os.path.join(data_dir, 'image', fname)
        mask_path = os.path.join(data_dir, 'label', fname)
        if ext == 'png' and os.path.isfile(mask_path):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
           
            if image.shape == mask.shape:
                crop_image(prefix, image, mask, save_dir)
                
                for scale in scale_range:
                    new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    new_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    new_prefix = '{}_{:.2f}'.format(prefix, scale)
                    crop_image(new_prefix, new_image, new_mask, save_dir)
            else:
                print("Shape is not match:", fname)

# training data
clear_data(config.train_dir)
process(config.raw_train_dir, config.train_dir, scale_range=[0.8])

# valdiation data
clear_data(config.val_dir)
process(config.raw_val_dir, config.val_dir, scale_range=[])