from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import segmentation_models as sm
from keras_radam import RAdam
from model import *
from data import *
import config
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_data_gen_args = dict(
                    # featurewise_center = False, 
                    # samplewise_center = False,
                    # rotation_range = 10, 
                    # width_shift_range = 0.01, 
                    # height_shift_range = 0.01, 
                    # shear_range = 0.01,
                    # zoom_range = [0.9, 1.1],  
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                )
val_data_gen_args = dict(
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                )

traning_generator = dataGenerator(config.batch_size, config.train_dir, 'image', 'label', train_data_gen_args, 
                                target_size=config.input_size[:-1], save_to_dir=config.save_gen_img,
                                seed=None)
validation_generator = dataGenerator(config.batch_size, config.val_dir, 'image', 'label', val_data_gen_args, 
                                target_size=config.input_size[:-1], save_to_dir=None,
                                seed=None)

N_TRAIN_SAMPLES = len(os.listdir(os.path.join(os.path.join(config.train_dir, 'image'))))
N_TEST_SAMPLES = len(os.listdir(os.path.join(os.path.join(config.val_dir, 'image'))))

print("\n-----------------------------------------------------------------")
print("\n-------------------------------RUN-------------------------------")
print("\n-----------------------------------------------------------------\n")

print("Number of training set:", N_TRAIN_SAMPLES)
print("Number of validation set:", N_TEST_SAMPLES)

# model = unet(input_size = config.input_size)
model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=config.input_size, classes=1, activation='sigmoid')
model.summary()

if(config.pretrained_weights):
    print("Load pretrained model!!!")
    model.load_weights(config.pretrained_weights)

model.compile(
    optimizer = Adam(lr=config.learning_rate), 
    # optimizer = RAdam(), 
    # loss = 'binary_crossentropy', 
    # loss = sm.losses.bce_dice_loss, 
    loss = sm.losses.bce_jaccard_loss, 
    metrics = [sm.metrics.iou_score]
)
model_checkpoint = ModelCheckpoint(config.checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)
lr_reduction = ReduceLROnPlateau(monitor='loss', 
                                patience=1, 
                                verbose=1, 
                                factor=0.3, 
                                min_lr=1e-8)
model.fit_generator(
    traning_generator, 
    steps_per_epoch=N_TRAIN_SAMPLES // config.batch_size,
    initial_epoch=config.initial_epoch,
    epochs=config.num_epoch, 
    validation_data=validation_generator,
    validation_steps=N_TEST_SAMPLES // config.batch_size,
    verbose=1,
    callbacks=[model_checkpoint, lr_reduction])
