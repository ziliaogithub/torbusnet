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
parser.add_argument('--max_dist', type=float, default=40, help='Ignore obstacles beyond this distance (meters)')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--optimizer', default='adam', help='adam or momentum ')

FLAGS = parser.parse_args()

BATCH_SIZE    = FLAGS.batch_size
NUM_POINT     = FLAGS.num_point
MAX_EPOCH     = FLAGS.max_epoch
LEARNING_RATE = FLAGS.learning_rate
OPTIMIZER     = FLAGS.optimizer
DATA_DIR      = FLAGS.data_dir
MAX_DIST      = FLAGS.max_dist

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
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters=2048, kernel_size=(1, 1), activation='relu')(p)

    p = MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')(p)
    p = Flatten()(p)
    p = Dense(512, activation='relu')(p)
    p = Dense(256, activation='relu')(p)
    c = Dense(3, activation=None)(p)
    s = Dense(3, activation=None)(p)
    centroids  = Lambda(lambda x: x * (90., 90., 2.))(c) # tx ty tz
    dimensions = Lambda(lambda x: x * (2., 90., 90.))(s) # h w l
    model = Model(inputs=points, outputs=[centroids, dimensions])
    model.compile(
        loss='mse',
        optimizer=Adam(lr=lr))
    return model

model = get_model_functional()
model.summary()

# -----------------------------------------------------------------------------------------------------------------
def gen(items, batch_size, num_points, training=True):
    lidars      = np.empty((batch_size, num_points, 4))
    centroids   = np.empty((batch_size, 3))
    dimensions  = np.empty((batch_size, 3))
    weights     = np.ones((batch_size))

    BINS = 40.

    if training is True:
        distances = []
        for item in items:
            tracklet, frame = item
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2])  # only x y
            distances.append(distance)

        h_density, h_edges = np.histogram(distances, bins=int(BINS), range=(0, MAX_DIST), density=True)
        h_count,   h_edges = np.histogram(distances, bins=int(BINS), range=(0, MAX_DIST), density=False)
        print(h_density)
        print(h_count)

    i = 0

    while True:
        random.shuffle(items)
        distances = []

        for item in items:
            tracklet, frame = item
            random_yaw   = np.random.random_sample() * 2. * np.pi - np.pi
            lidar        = tracklet.get_lidar(frame, num_points)[:, :4]
            centroid     = tracklet.get_box_centroid(frame)[:3]
            dimension    = tracklet.get_box_size()[:3]

            distance     = np.linalg.norm(centroid[:2]) # only x y
            distances.append(distance)

            if training is True:
                lidar        = point_utils.rotZ(lidar,    random_yaw)
                centroid     = point_utils.rotZ(centroid, random_yaw)
                # flip along x axis
                if np.random.randint(2) == 1:
                    lidar[:,0]  = -lidar[:,0]
                    centroid[0] = -centroid[0]
                # flip along y axis
                if np.random.randint(2) == 1:
                    lidar[:,1]  = -lidar[:,1]
                    centroid[1] = -centroid[1]


            lidars[i]     = lidar
            centroids[i]  = centroid
            dimensions[i] = dimension
            #print(distance)
            #print(int(BINS*distance/MAX_DIST))
            if training is True:
                weights[i]    = (1. / h_density[int(BINS*distance/MAX_DIST)]) #/ np.sum(h_density)
            #print( weights[i] )

            i += 1
            if i == batch_size:
                if training is True:
                    yield (lidars, [centroids, dimensions], [weights,weights])
                else:
                    yield (lidars, [centroids, dimensions])

                i = 0
        # we should be done here
        #count, edges = np.histogram(distances, bins=BINS, range=(0,MAX_DIST))
        #print()
        #print(count, edges)

def get_items(tracklets):
    items = []
    for tracklet in tracklets:
        for frame in tracklet.frames():
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2]) # only x y
            if distance < MAX_DIST:
                items.append((tracklet, frame))
    return items

train_items    = get_items(provider_didi.get_tracklets(DATA_DIR, "train.txt"))
validate_items = get_items(provider_didi.get_tracklets(DATA_DIR, "validate.txt"))

print("Train items:    " + str(len(train_items)))
print("Validate items: " + str(len(validate_items)))

#from sklearn.model_selection import train_test_split
#train_items, validation_items = train_test_split(items, test_size=0.20)

model.fit_generator(
    gen(train_items, BATCH_SIZE, NUM_POINT),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validate_items, BATCH_SIZE, NUM_POINT, training=False),
    validation_steps = len(validate_items) // BATCH_SIZE,
    epochs = 2000)