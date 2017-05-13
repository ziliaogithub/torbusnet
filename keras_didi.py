import provider_didi
import argparse
from keras.initializers import Constant, Zeros
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model, Input
from keras.layers import Input, merge, Layer
from keras.layers.merge import dot, Dot
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D, Cropping2D, AveragePooling2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

import os
import numpy as np
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'didi-competition/tracklets/python'))

from diditracklet  import *
import point_utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../release2/Data-points-processed', help='Tracklets top dir')
parser.add_argument('--num_point', type=int, default=27000, help='Number of lidar points to use')  #real number per lidar cycle is 32000, we will reduce to 16000
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--optimizer', default='adam', help='adam or momentum ')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
LEARNING_RATE = FLAGS.learning_rate
OPTIMIZER = FLAGS.optimizer
DATA_DIR = FLAGS.data_dir

# -----------------------------------------------------------------------------------------------------------------
def get_model(lr=LEARNING_RATE):
    model = Sequential()
    model.add(Lambda(lambda x: x * (1 / 90., 1 / 90., 1 / 2., 1 / 127.5) - (0., 0., 0., 1.),
                     input_shape=(NUM_POINT, 4),
                     output_shape=(NUM_POINT, 4)))
    model.add(Reshape(target_shape=(NUM_POINT, 4, 1), input_shape=(NUM_POINT, 4)))
    model.add(Conv2D(filters=64, kernel_size=(1, 4), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(filters=1024, kernel_size=(1, 1), activation='relu'))

    model.add(MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation=None))
    model.add(Lambda(lambda x: x * (90., 90., 2.)))
    model.compile(
        loss='mse',
        optimizer=Adam(lr=lr))
    return model

def get_model_functional(lr=LEARNING_RATE):
    points = Input(shape=(NUM_POINT, 4))

    p = Lambda(lambda x: x * (1 / 90., 1 / 90., 1 / 2., 1 / 127.5) - (0., 0., 0., 1.))(points)
    p = Reshape(target_shape=(NUM_POINT, 4, 1), input_shape=(NUM_POINT, 4))(p)
    p = Conv2D(filters=  64, kernel_size=(1, 4), activation='relu')(p)
    p = Conv2D(filters=  64, kernel_size=(1, 1), activation='relu')(p)
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # feature transform
    K = 64
    f = Conv2D(filters=  64, kernel_size=(1, 1), activation='relu')(p)
    f = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(f)
    f = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu')(f)
    f = MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')(f)
    f = Flatten()(f)
    f = Dense(512, activation='relu')(f)
    f = Dense(256, activation='relu')(f)
    f = Dense(K * K, activation='relu')(f)
    f = Reshape(target_shape=(K, K))(f)

    p = Reshape(target_shape=(NUM_POINT, K))(p)
    p = Dot(axes=(2))([p,f])
    p = Reshape(target_shape=(NUM_POINT,1, K))(p)
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    p = Conv2D(filters=  64, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu')(p)

    p = MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')(p)
    p = Flatten()(p)
    p = Dense(512, activation='relu')(p)
    p = Dense(256, activation='relu')(p)
    p = Dense(3, activation=None)(p)
    centroids = Lambda(lambda x: x * (90., 90., 2.))(p)
    model = Model(inputs=points, outputs=centroids)
    model.compile(
        loss='mse',
        optimizer=Adam(lr=lr))
    return model

model = get_model_functional()
model.summary()

# -----------------------------------------------------------------------------------------------------------------
def gen(items, batch_size, num_points, validation=False):
    lidars      = np.empty((batch_size, num_points, 4))
    centroids   = np.empty((batch_size, 3))
    i = 0

    while True:
        random.shuffle(items)
        for item in items:
            tracklet, frame = item
            random_yaw   = np.random.random_sample() * 2. * np.pi - np.pi
            lidar        = tracklet.get_lidar(frame, num_points)[:, :4]
            centroid     = tracklet.get_box_centroid(frame)[:3]

            if validation is False:
                lidar        = point_utils.rotZ(lidar,    random_yaw)
                centroid     = point_utils.rotZ(centroid, random_yaw)
            lidars[i]    = lidar
            centroids[i] = centroid

            i += 1
            if i == batch_size:
                yield (lidars, centroids)
                i = 0

tracklets = provider_didi.get_tracklets(os.path.join(DATA_DIR))
items = []
for tracklet in tracklets:
    for frame in tracklet.frames():
        items.append((tracklet, frame))

from sklearn.model_selection import train_test_split

train_items, validation_items = train_test_split(items, test_size=0.20)

model.fit_generator(
    gen(train_items, BATCH_SIZE, NUM_POINT),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validation_items, BATCH_SIZE, NUM_POINT, validation=True),
    validation_steps = len(validation_items) // BATCH_SIZE,
    epochs = 2000)