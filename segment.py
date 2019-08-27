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
    position, input_data = split(image, strides=(50, 50))
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
        prediction[prediction == 0] = 1e-9
        for i in range(len(position)):
            x, y = position[i]
            total_mask[x : x + config.input_size[0], y : y + config.input_size[1]] += np.log(prediction[i,:,:,0])
            count_mask[x : x + config.input_size[0], y : y + config.input_size[1]] += 1
        return np.exp(total_mask / count_mask)
    else:
        print("No mode fought for ensembling!")
        return None
    

def evaluate(image_path, mask_path, mode, thresholds=[]):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    image = np.array(image) / 255
    mask = np.array(mask) / 255
    
    mask_proba = get_mask(image, mode=mode)

    # save result
    # prefix = image_path.split('/')[-1].split('.')[0]
    # cv2.imwrite('test/' + prefix + '_' + mode + '_mask_proba.png', mask_proba * 255)
    # mask_proba = (mask_proba > 0.5).astype(np.uint8)
    # cv2.imwrite('test/' + prefix + '_' + mode + '_mask.png', mask_proba * 255)

    score = iou_score([mask], [mask_proba], threshold=None)
    thres_scores = []
    for threshold in thresholds:
        iou = iou_score([mask], [mask_proba], threshold=threshold)
        thres_scores.append(iou)
    return score, thres_scores

def get_best_threshold():
    scores = []
    thres_scores = []
    thresholds = np.arange(0.1, 0.91, 0.05)
    for fname in os.listdir(os.path.join(config.raw_val_dir, 'image')):
        score, thres_score = evaluate(os.path.join(config.raw_val_dir, 'image', fname), os.path.join(config.raw_val_dir, 'label', fname), 'average', thresholds)
        scores.append(score)
        thres_scores.append(thres_score)
        
    scores = np.mean(scores)
    thres_scores = np.mean(thres_scores, axis=0)
    thres_scores = np.around(thres_scores, 4)
    print("Score: {:.3f}".format(scores))
    print("Smooth score with threshold")
    print("Threshold -> Score")
    for i in range(len(thresholds)):
        print("{:.2f} -> {:.3f}".format(thresholds[i], thres_scores[i]))

    print("-----Conclusion-----")
    max_value = np.max(thres_scores)
    max_range = np.where(np.array(thres_scores) == max_value)[0]
    print("Max score: {:.3f}, threshold: [{:.2f}, {:.2f}]".format(max_value, thresholds[max_range[0]], thresholds[max_range[-1]]))

get_best_threshold()

# evaluate('dataset/raw/train/image/III-I7.png', 'dataset/raw/train/label/III-I7.png', mode = 'average')
# evaluate('dataset/raw/train/image/III-I7.png', 'dataset/raw/train/label/III-I7.png', mode = 'geometric')

# evaluate('dataset/raw/val/image/I-I7.png', 'dataset/raw/val/label/I-I7.png', mode = 'average')
# evaluate('dataset/raw/val/image/I-I7.png', 'dataset/raw/val/label/I-I7.png', mode = 'geometric')




# data_dir = 'test/'
# for fname in os.listdir(data_dir):
#     if 'mask.png' in fname:
#         prefix = fname.split('_')[0]
#         image = cv2.imread(os.path.join(config.raw_val_dir, 'image', prefix + ".png"))
#         if image is None:
#             image = cv2.imread(os.path.join(config.raw_train_dir, 'image', prefix + ".png"))

#         mask = cv2.imread(os.path.join(data_dir, prefix + "_average_mask.png"))
        
#         concat = np.concatenate((image, mask), axis=0)
#         cv2.imwrite(os.path.join(data_dir, 'concat', prefix + '_concat.png'), concat)


