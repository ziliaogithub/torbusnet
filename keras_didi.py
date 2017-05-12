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
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'didi-competition/tracklets/python'))

from diditracklet  import *


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

# -----------------------------------------------------------------------------------------------------------------
model = Sequential()
model.add(Lambda(lambda x: x * (1/90.,1/90.,1/2., 1/127.5) - (0.,0.,0.,1.),
            input_shape=(NUM_POINT, 4),
            output_shape=(NUM_POINT, 4)))
model.add(Reshape(target_shape=(NUM_POINT, 4,1),input_shape=(NUM_POINT, 4)))
model.add(Conv2D(filters=  64, kernel_size=(1,4), activation='relu'))
model.add(Conv2D(filters=  64, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(filters=  64, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(filters= 128, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(filters=1024, kernel_size=(1,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation=None))
model.add(Lambda(lambda x: x * (90.,90.)))
model.compile(
    loss='mse',
	optimizer=Adam(lr=1e-4))
model.summary()

# -----------------------------------------------------------------------------------------------------------------
def gen(tracklets, batch_size, num_points):
    lidars      = np.empty((batch_size, num_points, 4))
    centroids   = np.empty((batch_size, 2))
    i = 0

    items = []
    for tracklet in tracklets:
        for frame in tracklet.frames():
            items.append((tracklet, frame))

    while True:
        random.shuffle(items)
        for item in items:
            tracklet, frame = item
            lidars[i]    = tracklet.get_lidar(frame, num_points)[:, :4]
            centroids[i] = tracklet.get_box_centroid(frame)[:2]
            i += 1
            if i == batch_size:
                yield (lidars, centroids)
                i = 0

tracklets              = provider_didi.get_tracklets(os.path.join(DATA_DIR))
total_number_of_frames = sum([len(t.frames()) for t in tracklets])

model.fit_generator(gen(tracklets, BATCH_SIZE, NUM_POINT), steps_per_epoch = total_number_of_frames // BATCH_SIZE, epochs = 200)