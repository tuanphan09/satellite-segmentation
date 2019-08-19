import string

#------------------------Prepare Data config------------------------#

raw_train_dir = 'dataset/raw/train'
train_dir = 'dataset/aug/train'
raw_val_dir = 'dataset/raw/val'
val_dir = 'dataset/aug/val'

input_size = (224, 400, 3)
strides = (100, 200)

#------------------------Training config------------------------#
learning_rate = 1e-4
batch_size = 3
checkpoint_path = 'models/weights.{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5' 

initial_epoch = 3
pretrained_weights = ''

num_epoch = 30

save_gen_img = None
# save_gen_img = 'dataset/gen'




