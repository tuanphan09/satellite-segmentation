from __future__ import print_function

import numpy as np 
import os
import glob
import cv2
import random
import string
import keras

import config

class SatelliteDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, list_labels, batch_size=4, save_to_dir=None, is_testing=False):
        'Initialization'
        
        self.list_files = list_files
        self.list_labels = list_labels
        self.batch_size = batch_size
        self.save_to_dir = save_to_dir
        self.is_testing = is_testing
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.is_testing:
            return int(np.floor((len(self.list_files)-1) / self.batch_size + 1))
        return int(np.floor(len(self.list_files) / self.batch_size))
        

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_files_batch = [self.list_files[k] for k in indexes]
        list_labels_batch = [self.list_labels[k] for k in indexes]

        # Generate data
        images, masks = self.__data_generation(list_files_batch, list_labels_batch)
        return images, masks

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.is_testing == False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_batch, list_labels_batch):
        'Generates data containing batch_size samples' 
        
        images = []
        masks = []

        for i in range(len(list_files_batch)):
            image = cv2.imread(list_files_batch[i])
            mask = cv2.imread(list_labels_batch[i])
            assert image.shape == mask.shape

            image_crop = None
            mask_crop = None
            for i in range(3):
                x, y = np.random.randint(low=0, high=[image.shape[0] - config.input_size[0], image.shape[1] - config.input_size[1]], size=2)
                
                image_crop = image[x: x+config.input_size[0], y: y+config.input_size[1], :]
                mask_crop = mask[x: x+config.input_size[0], y: y+config.input_size[1], :]
                if np.sum((mask_crop > 200).astype(np.uint8)) / 3 >= config.min_num_pixel:
                    break
            
            if self.save_to_dir is not None:
                if self.is_testing:
                    subdir = 'val'
                else:
                    subdir = 'train'
                fname = ''.join(random.sample(string.ascii_lowercase, 10)) + '.png'
                cv2.imwrite(os.path.join(self.save_to_dir, subdir, fname), np.concatenate((image_crop, mask_crop), axis=1))

            mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)
            mask_crop = np.expand_dims(mask_crop, axis=-1)
            
            images.append(image_crop)
            masks.append(mask_crop)
            

        return self.normalize(images, masks)
    
    def normalize(self, img, mask):
        img = np.array(img)
        mask = np.array(mask)
        if(np.max(img) > 1):
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return img, mask