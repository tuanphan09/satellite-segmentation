from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import segmentation_models as sm
from keras_radam import RAdam
from model import *
import config
from data_generator import SatelliteDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_data(data_dir):
    list_files = [] 
    list_labels = []

    for fname in os.listdir(os.path.join(data_dir, 'image')):
        if os.path.exists(os.path.join(data_dir, 'label', fname)):
            list_files.append(os.path.join(data_dir, 'image', fname))
            list_labels.append(os.path.join(data_dir, 'label', fname))
    return list_files, list_labels

list_files_train, list_labels_train = get_data(config.raw_train_dir)
list_files_val, list_labels_val = get_data(config.raw_val_dir)


training_generator = SatelliteDataGenerator(list_files_train, list_labels_train, batch_size=config.batch_size, save_to_dir=config.save_gen_img, is_testing=False)
validation_generator = SatelliteDataGenerator(list_files_val, list_labels_val, batch_size=config.batch_size, save_to_dir=config.save_gen_img, is_testing=True)



print("\n-----------------------------------------------------------------")
print("-------------------------------RUN-------------------------------")
print("-----------------------------------------------------------------\n")

N_TRAIN_SAMPLES = len(list_files_train)
N_TEST_SAMPLES = len(list_files_val)
print("Number of training set:", N_TRAIN_SAMPLES)
print("Number of validation set:", N_TEST_SAMPLES)

model = unet(input_size = config.input_size)
# model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=config.input_size, classes=1, activation='sigmoid')
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
                                verbose=3, 
                                factor=0.5, 
                                min_lr=1e-8)
model.fit_generator(
    training_generator, 
    steps_per_epoch=5,#N_TRAIN_SAMPLES // config.batch_size,
    initial_epoch=config.initial_epoch,
    epochs=config.num_epoch, 
    validation_data=validation_generator,
    validation_steps=N_TEST_SAMPLES // config.batch_size,
    verbose=1,
    callbacks=[model_checkpoint, lr_reduction])
