import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
from segmentation_models.losses import dice_loss, jaccard_loss

SMOOTH = 1.

# keras loss for training

def bce_log_dice_loss(gt, pr, smooth=SMOOTH, per_image=True, beta=1.):
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce - K.log(dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=beta))
    return loss

def bce_log_jaccard_loss(gt, pr, smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce - K.log(jaccard_loss(gt, pr, smooth=smooth, per_image=per_image))
    return loss





# metric for final result

def iou_score(gt, pr, smooth=SMOOTH, per_image=True, threshold=None):
    gt = np.array(gt, dtype=np.float)
    pr = np.array(pr, dtype=np.float)
    assert len(gt.shape) == len(pr.shape) == 3

    if per_image:
        axes = (1, 2)
    else:
        axes = (0, 1, 2)
        
    if threshold is not None:
        pr = (pr > threshold).astype(np.float)

    intersection = np.sum(gt * pr, axis=axes)
    union = np.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = np.mean(iou, axis=0)

    return iou


# ensemble strategy for final result
def gemetric_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
def gemetric_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))
