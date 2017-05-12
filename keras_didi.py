import provider_didi
import argparse
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D, Cropping2D, AveragePooling2D
from keras.activations import relu
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../release2/Data-points-processed', help='Tracklets top dir')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [256/512/1024/2048] [default: 1024]')  #real number per lidar cycle is 32000, we will reduce to 16000
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
OPTIMIZER = FLAGS.optimizer
DATA_DIR = FLAGS.data_dir
data = provider_didi.get_tracklets(os.path.join(DATA_DIR))

idxs = np.arange(0, len(data))
np.random.shuffle(idxs)
# Split into test and training set (20:80)
split =  int(0.8*len(data))
TRAIN_FILES =  data[:split]
TEST_FILES  =  data[split:]

# -----------------------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Reshape(target_shape=(NUM_POINT, 3,1),input_shape=(NUM_POINT, 3)))
model.add(Conv2D(filters=  64, kernel_size=(1,3), activation='relu'))
model.add(Conv2D(filters=  64, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(filters=  64, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(filters= 128, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(filters=1024, kernel_size=(1,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation=None))
model.compile(
    loss='mse',
	optimizer=Adam(lr=1e-4))
model.summary()



# -----------------------------------------------------------------------------------------------------------------


#exception when len(data) == 1
if split == 0:
    TRAIN_FILES = TEST_FILES

train_file_idxs = np.arange(0, len(TRAIN_FILES))
#np.random.shuffle(train_file_idxs)

for fn in range(len(TRAIN_FILES)):
    current_data, current_label = provider_didi.loadDataFile(TRAIN_FILES[train_file_idxs[fn]],NUM_POINT)
    #current_data, current_label, _ = provider_didi.shuffle_data(current_data, current_label)

    file_size = current_data.shape[0]
    num_batches = int(file_size / BATCH_SIZE)

    model.fit(current_data, current_label,
              epochs=200,
              batch_size=8)
