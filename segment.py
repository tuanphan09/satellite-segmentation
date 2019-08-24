import cv2
import numpy
import os
import random
from utils import *
from model import *
import config
import segmentation_models as sm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load_pretrained_model():
    model = unet(input_size = config.input_size)
    # model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=config.input_size, classes=1, activation='sigmoid')
    if(config.pretrained_weights):
        print("\n......Loading pretrained model from {}....\n".format(config.pretrained_weights))
        model.load_weights(config.pretrained_weights)
    return model

model = load_pretrained_model()

# def blend(img, mask, color):
    # img = img[:,:,0] if len(img.shape) == 3 else img
    # img_out = np.zeros(img.shape + (3,))
    # for i in range(num_class):
    #     img_out[img == i,:] = color
    # return img_out / 255

def segmentaion_dir(data_dir):
    input_data = []
    for fname in os.listdir(data_dir):
        if random.randint(0, 100) < 10:
            image = cv2.imread(os.path.join(data_dir, fname))
            input_data.append(image)
            if len(input_data) == 20:
                break

    input_data = np.array(input_data)
    masks = model.predict(input_data / 255, verbose=1)
    # masks = (masks > 0.5).astype(np.uint8)
    masks *= 255

    for i in range(len(masks)):
        cv2.imwrite('test/' + str(i) + '_origin.png', input_data[i])
        cv2.imwrite('test/' + str(i) + '_mask.png', masks[i])

def split(image, strides=(100, 100)):
    position, input_image = [], []
    y = 0
    mark_col = 0
    while y + config.input_size[1] <= image.shape[1] or mark_col == 1:
        x = 0
        mark_row = 0
        while x + config.input_size[0] <= image.shape[0] or mark_col == 1:
            img_crop = image[x : x + config.input_size[0], y : y + config.input_size[1], :]
            position.append((x, y))
            input_image.append(img_crop)
            x += strides[0]
            if mark_row == 1:
                break
            if x + config.input_size[0] > image.shape[0] and x + config.input_size[0] != image.shape[0] + strides[0]:
                x = image.shape[0] - config.input_size[0]
                mark_row = 1

        y += strides[1]
        if mark_col == 1:
            break
        if y + config.input_size[1] > image.shape[1] and y + config.input_size[1] != image.shape[1] + strides[1]:
            y = image.shape[1] - config.input_size[1]
            mark_col = 1

    return position, input_image

def get_mask(image, mode='average'):
    position, input_data = split(image, strides=(100, 120))
    input_data = np.array(input_data)
    prediction = model.predict(input_data, verbose=1)

    total_mask = np.zeros(image.shape[:-1])    
    count_mask = np.zeros(image.shape[:-1])    
    if mode == 'average':
        for i in range(len(position)):
            x, y = position[i]
            total_mask[x : x + config.input_size[0], y : y + config.input_size[1]] += prediction[i,:,:,0]
            count_mask[x : x + config.input_size[0], y : y + config.input_size[1]] += 1
        return total_mask / count_mask
    elif mode == 'geometric':
        for i in range(len(position)):
            x, y = position[i]
            total_mask[x : x + config.input_size[0], y : y + config.input_size[1]] += np.log(prediction[i,:,:,0])
            count_mask[x : x + config.input_size[0], y : y + config.input_size[1]] += 1
        return np.exp(total_mask / count_mask)
    else:
        print("No mode fought for ensembling!")
        return None
    

def evaluate(image_path, mask_path, mode):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    image = np.array(image) / 255
    mask = np.array(mask) / 255
    
    mask_proba = get_mask(image, mode=mode)

    prefix = image_path.split('/')[-1].split('.')[0]

    cv2.imwrite('test/' + prefix + '_' + mode + '_mask.png', mask_proba * 255)
    mask_proba = (mask_proba > 0.5).astype(np.uint8)
    cv2.imwrite('test/' + prefix + '_' + mode + '_mask_thres.png', mask_proba * 255)

    # for threshold in np.arange(0.3, 0.9, 0.05):
    #     iou_score = iou_metric(mask, np.int32(mask_proba > threshold))
    #     print("{:.2f} -> {}".format(threshold, iou_score))


evaluate('dataset/raw/train/image/I1.png', 'dataset/raw/train/label/I1.png', mode = 'average')
evaluate('dataset/raw/train/image/I1.png', 'dataset/raw/train/label/I1.png', mode = 'geometric')


evaluate('dataset/raw/val/image/I61.png', 'dataset/raw/val/label/I61.png', mode = 'average')
evaluate('dataset/raw/val/image/I61.png', 'dataset/raw/val/label/I61.png', mode = 'geometric')
# for fname in os.listdir('dataset/raw/val/image'):
#     print(fname)
#     evaluate('dataset/raw/val/image/' + fname, 'dataset/raw/val/label/' + fname)





# for fname in os.listdir('dataset/raw/train/image'):
#     segmentaion_file(model, os.path.join('dataset/raw/train/image/', fname))
#     mask = proba_mask * 255
#     prefix = path.split('/')[-1].split('.')[0]
#     cv2.imwrite('test/large/train/' + prefix + '_image.png', image)
#     cv2.imwrite('test/large/train/' + prefix + '_mask.png', mask)

#     mask = (proba_mask > 0.5).astype(np.uint8) * 255
#     cv2.imwrite('test/large/train/' + prefix + '_mask_hard.png', mask)
# data_dir = 'test/large/train'
# for fname in os.listdir(data_dir):
#     if 'hard' in fname:
#         prefix = fname.split('_')[0]
#         image = cv2.imread(os.path.join(data_dir, prefix + ".png"))
#         mask = cv2.imread(os.path.join(data_dir, prefix + "_mask_hard.png"))
        
#         concat = np.concatenate((image, mask), axis=0)
#         cv2.imwrite(os.path.join(data_dir, prefix + '_concat.png'), concat)


