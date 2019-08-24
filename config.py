import string

#------------------------Prepare Data config------------------------#

raw_train_dir = 'dataset/raw/train'
train_dir = 'dataset/aug/train'
raw_val_dir = 'dataset/raw/val'
val_dir = 'dataset/aug/val'

input_size = (256, 320, 3)
strides = (200, 250)
min_num_pixel = 50

#------------------------Training config------------------------#
learning_rate = 1e-5
batch_size = 2
checkpoint_path = 'models/weights.{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5' 

initial_epoch = 0
pretrained_weights = '' #'models/weights.12-0.160-0.181.hdf5'

num_epoch = 50

save_gen_img = None
# save_gen_img = 'dataset/gen'




