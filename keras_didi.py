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
from keras.callbacks import ModelCheckpoint

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
parser.add_argument('--num_point', type=int, default=24000, help='Number of lidar points to use')  #real number per lidar cycle is 32000, we will reduce to 16000
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run')
parser.add_argument('--max_dist', type=float, default=25, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('--max_dist_offset', type=float, default=3, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--optimizer', default='adam', help='adam or momentum ')

FLAGS = parser.parse_args()

BATCH_SIZE     = FLAGS.batch_size
NUM_POINT      = FLAGS.num_point
MAX_EPOCH      = FLAGS.max_epoch
LEARNING_RATE  = FLAGS.learning_rate
OPTIMIZER      = FLAGS.optimizer
DATA_DIR       = FLAGS.data_dir
MAX_DIST       = FLAGS.max_dist

MAX_LIDAR_DIST = MAX_DIST + FLAGS.max_dist_offset

def get_model_functional(lr=LEARNING_RATE):
    points = Input(shape=(NUM_POINT, 4))
    MAX_XY = 90.
    MAX_Z  = 2.

    p = Lambda(lambda x: x * (1. / MAX_XY, 1. / MAX_XY, 1. / MAX_Z, 1 / 127.5) - (0., 0., 0., 1.))(points)
    p = Reshape(target_shape=(NUM_POINT, 4, 1), input_shape=(NUM_POINT, 4))(p)
    p = Conv2D(filters=  64, kernel_size=(1, 4), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters=2048, kernel_size=(1, 1), activation='relu')(p)

    p = MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')(p)
    p = Flatten()(p)
    p = Dense(512, activation='relu')(p)
    p = Dropout(0.3)(p)
    p = Dense(256, activation='relu')(p)
    p = Dropout(0.3)(p)
    c = Dense(3, activation=None)(p)
    s = Dense(3, activation=None)(p)
    centroids  = Lambda(lambda x: x * (MAX_XY, MAX_XY, MAX_Z))(c) # tx ty tz
    dimensions = Lambda(lambda x: x * (MAX_Z, MAX_XY, MAX_XY))(s) # h w l
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

    BINS = 25.

    if training is True:
        # our training set is heavily unbalanced, so we are going to balance the histogram of x,y labels.
        # since we will be using rotations we will build a target histogram based on distance so that
        # the probability of a label ending up an a given x,y bucket is roughly the same
        # we build a distance histogram target and since the number of points sampled from the a distance
        # needs to grow proportionally with the radius of the circumference, we factor it in.
        distances = []
        for item in items:
            tracklet, frame = item
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2])  # only x y
            distances.append(distance)

        h_count,   _h_edges = np.histogram(distances, bins=int(BINS), range=(0, MAX_DIST), density=False)
        h_edges = np.empty(_h_edges.shape[0]-1)
        i = range(h_edges.shape[0])
        for i in range(h_edges.shape[0]):
            h_edges[i] = (_h_edges[i] + _h_edges[i+1] ) / 2.
        h_edges[h_count == 0] = 0
        print(h_edges)
        print(h_count)
        h_target = None
        for i,r in enumerate(h_edges):
            if r > 0:
                k = h_count[i]/r
                _h_target = np.array(h_edges * k, dtype=np.int32)
                print(_h_target)
                if np.all(h_count - _h_target >= 0):
                    h_target = _h_target
        if h_target is None:
            print("WARNING: Could not find optimal balacing set, reverting to bad one")
            h_target[h_count > 0] = np.amin(h_count[h_count > 0])
        h_current = h_target.copy()

        print("Target distance histogram", h_target)
    else:
        skip = False

    i = 0

    while True:
        random.shuffle(items)

        for item in items:
            tracklet, frame = item

            centroid     = tracklet.get_box_centroid(frame)[:3]
            dimension    = tracklet.get_box_size()[:3]
            distance     = np.linalg.norm(centroid[:2]) # only x y

            if training is True:

                _h = int(BINS * distance / MAX_DIST)

                if h_current[_h] == 0:
                    skip = True
                else:
                    skip = False
                    h_current[_h] -= 1
                    if np.sum(h_current) == 0:
                        h_current = h_target.copy()

                    lidar = tracklet.get_lidar(frame, num_points, max_distance=MAX_LIDAR_DIST)[:, :4]

                    # random rotation along Z axis
                    random_yaw = np.random.random_sample() * 2. * np.pi - np.pi
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

                    # jitter cloud and intensity and move everything a random amount
                    random_t = np.random.random_sample((4)) * np.array([MAX_DIST/BINS, MAX_DIST/BINS, 0.5, 0.]) \
                               - np.array([MAX_DIST/(BINS*2), MAX_DIST/(BINS*2), 0.25, 0.])
                    lidar        += random_t + np.random.random_sample((lidar.shape[0], 4)) * np.array([0.04, 0.04, 0.04, 4.]) - np.array([0.02, 0.02, 0.02, 2.])
                    centroid[:3] += random_t[:3]

            else:
                lidar = tracklet.get_lidar(frame, num_points, max_distance=(MAX_DIST + 3.))[:, :4]

            if skip is False:
                lidars[i]     = lidar
                centroids[i]  = centroid
                dimensions[i] = dimension

                i += 1
                if i == batch_size:
                    yield (lidars, [centroids, dimensions])
                    i = 0

def get_items(tracklets):
    items = []
    for tracklet in tracklets:
        for frame in tracklet.frames():
            state    = tracklet.get_state(frame)
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2]) # only x y
            if (distance < MAX_DIST) and (state == 1):
                items.append((tracklet, frame))
    return items

train_items    = get_items(provider_didi.get_tracklets(DATA_DIR, "train.txt"))
validate_items = get_items(provider_didi.get_tracklets(DATA_DIR, "validate.txt"))

print("Train items:    " + str(len(train_items)))
print("Validate items: " + str(len(validate_items)))

    #from sklearn.model_selection import train_test_split
#train_items, validation_items = train_test_split(items, test_size=0.20)
save_checkpoint = ModelCheckpoint(
    "torbusnet-epoch{epoch:02d}-loss{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit_generator(
    gen(train_items, BATCH_SIZE, NUM_POINT),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validate_items, BATCH_SIZE, NUM_POINT, training=False),
    validation_steps = len(validate_items) // BATCH_SIZE,
    epochs = 2000,
    callbacks = [save_checkpoint])