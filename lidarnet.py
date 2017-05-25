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
from keras.optimizers import Adam, Nadam
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from torbus_layers import TorbusMaxPooling2D
import multi_gpu

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
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run')
parser.add_argument('--max_dist', type=float, default=25, help='Ignore centroids beyond this distance (meters)')
#parser.add_argument('--max_dist_offset', type=float, default=3, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('-b', '--batch_size', type=int, nargs='+', default=[12], help='Batch Size during training, or list of batch sizes for each GPU, e.g. -b 12,8')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs used for training')

args = parser.parse_args()

MAX_EPOCH      = args.max_epoch
LEARNING_RATE  = args.learning_rate
DATA_DIR       = args.data_dir
MAX_DIST       = args.max_dist
#MAX_LIDAR_DIST = MAX_DIST + args.max_dist_offset

RINGS = range(11,20)
POINTS_PER_RING = 2048

assert args.gpus  == len(args.batch_size)

def get_model():
    # receptive field needs to be 1/5 of total to detect large objects close by (e.g. for 1024 -> 205)
    NRINGS = len(RINGS)
    #ring_points = Input(shape=(RINGS, POINTS_PER_RING, 3)) # x y z
    ring_d_i    = Input(shape=(NRINGS, POINTS_PER_RING, 2)) # d i

    # assume d  90 max =>  90 -> 1, 0 -> -1
    #        i 128 max => 128 -> 1, 0 -> -1
    d_i = Lambda(lambda x: x * (1/50. , 1. / 64.) - (1., 1.))(ring_d_i)
   # d_i = Lambda(lambda x: x * (1/50.) - (1.))(ring_d_i)


#    d_i = Lambda(lambda x: x * (1. / 45., 1. / 64.) - (1., 1.))(ring_d_i)
    print(d_i)
    act = 'tanh'

    p = Conv2D(filters=    8, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(d_i) #  receptive 3
    p = Conv2D(filters=   16, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(p) #   5
    p = MaxPooling2D(pool_size=(1,2), strides=None, padding='valid')(p)
    p = Conv2D(filters=   32, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(p) #   9
    #p = MaxPooling2D(pool_size=(1,2), strides=None, padding='valid')(p)
    p = Conv2D(filters=   32, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(p) #  17\    p = MaxPooling2D(pool_size=(1,2), strides=None, padding='valid')(p)
    #p = MaxPooling2D(pool_size=(1,2), strides=None, padding='valid')(p)
    p = Conv2D(filters=   32, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(p) #  33\    p = MaxPooling2D(pool_size=(1,2), strides=None, padding='valid')(p)
    #p = MaxPooling2D(pool_size=(1,2), strides=None, padding='valid')(p)
    p = Conv2D(filters=   32, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(p) #  65
    p = Conv2D(filters=   32, kernel_size=(NRINGS, 5), dilation_rate=(1, 1), activation=act, padding='same')(p) # 129
    #p = Conv2D(filters=  128, kernel_size=(NRINGS, 3), dilation_rate=(1,64), activation=act, padding='same')(p) # 257
    p = Conv2D(filters=   32, kernel_size=(NRINGS, 1), dilation_rate=(1, 1), activation=act, padding='same')(p) # 257
    #p = Conv2D(filters=   32, kernel_size=(NRINGS, 1), dilation_rate=(1, 1), activation=act, padding='same')(p) # 257
    #p = Conv2D(filters=   16, kernel_size=(NRINGS, 1), dilation_rate=(1, 1), activation=act, padding='same')(p) # 257

    p = MaxPooling2D(pool_size=(1,16), strides=None, padding='valid')(p)
    # at this point we have activations for each angle and ring, now learn to combine them.

    p = Flatten()(p)
    p = Dense(64, activation=act)(p)
    p = Dense(32, activation=act)(p)
    angle = Dense(1)(p)

    # p is RINGSx3 activations per
    # c = Multiply([p, ring_points])
    #centroids = MaxPooling2D(pool_size=(REGRESSION_POINTS, 1), strides=None, padding='valid')

    #centroids  = Lambda(lambda x: x * (25.,25., 3.) - (0., 0., -1.5))(c) # tx ty tz
    #dimensions = Lambda(lambda x: x * ( 3.,25.,25.) - (-1.5, 0., 0.))(s) # h w l
    model = Model(inputs=ring_d_i, outputs=angle)
    return model

if args.model:
    print("Loading model " + args.model)
    model = load_model(args.model)
    model.summary()
else:
    model = get_model()
    model.summary()

if (args.gpus > 1) or (len(args.batch_size) > 1):
    if len(args.batch_size) == 1:
        model = multi_gpu.make_parallel(model, args.gpus )
        BATCH_SIZE = args.gpus * args.batch_size[0]
    else:
        BATCH_SIZE = sum(args.batch_size)
        model = multi_gpu.make_parallel(model, args.gpus , splits=args.batch_size)
else:
    BATCH_SIZE = args.batch_size[0]

model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

# -----------------------------------------------------------------------------------------------------------------
def gen(items, batch_size, points_per_ring = POINTS_PER_RING, training=True, rings = RINGS):
    lidar_d_i_s    = np.empty((batch_size, len(rings), points_per_ring, 1))
    angles         = np.empty((batch_size, 1))

    BINS = 25.

    PAD = 0

    if training is True:

        # our training set is heavily unbalanced, so we are going to balance the histogram of x,y labels.
        # since we will be using rotations we will build a target histogram based on distance so that
        # the probability of a label ending up an a given x,y bucket would be roughly the same
        # we build a distance histogram target and since the number of points sampled from a distance
        # needs to grow proportionally with the radius of the circumference, we factor it in.
        distances = []
        for item in items:
            tracklet, frame = item
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2])  # only x y
            distances.append(distance)

        h_count,  _h_edges = np.histogram(distances, bins=int(BINS), range=(0, MAX_DIST), density=False)
        h_edges = np.empty(_h_edges.shape[0]-1)

        for i in range(h_edges.shape[0]):
            h_edges[i] = (_h_edges[i] + _h_edges[i+1] ) / 2.

        h_edges[h_count == 0] = 0
        h_target = None
        best_min = -1
        for i,r in enumerate(h_edges):
            if r > 0:
                k = h_count[i]/r
                _h_target = np.array(h_edges * k, dtype=np.int32)
                if np.all(h_count - _h_target >= 0):
                    _min = np.amin(h_count[_h_target > 0] - _h_target[_h_target > 0])
                    # todo: improve when there are best candidates
                    if _min > best_min:
                        h_target = _h_target
                        best_min = _min
        if h_target is None:
            print("WARNING: Could not find optimal balacing set, reverting to bad one")
            h_target[h_count > 0] = np.amin(h_count[h_count > 0])
        h_current = h_target.copy()

        print("Target distance histogram", h_target)
    else:
        skip = False

    i = 0
    seen = 0
    xyhist = np.zeros((600,600), dtype=np.float32)

    while True:
        random.shuffle(items)

        for item in items:
            tracklet, frame = item

            centroid     = tracklet.get_box_centroid(frame)[:3]
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

                    # random rotation along Z axis
                    random_yaw = (np.random.random_sample() * 2. - 1.) * np.pi
                    centroid = point_utils.rotZ(centroid, random_yaw)
                    if False:
                        # flip along x axis
                        if np.random.randint(2) == 1:
                            lidar[:, 0] = -lidar[:, 0]
                            centroid[0] = -centroid[0]
                        # flip along y axis
                        if np.random.randint(2) == 1:
                            lidar[:, 1] = -lidar[:, 1]
                            centroid[1] = -centroid[1]
                    lidar_d_i = tracklet.get_lidar_rings(frame, rings = rings, points_per_ring = points_per_ring, pad = PAD, diff=False, rotate = random_yaw) #
                    #print(lidar_d_i[5,s:s+10,0])

            else:
                lidar_d_i = tracklet.get_lidar_rings(frame, rings = rings, points_per_ring = points_per_ring, pad = PAD, diff=False)  #

            if skip is False:


                lidar_d_i_s[i]  = lidar_d_i[...,0:1]
                angles[i]       = np.arctan2(centroid[1], centroid[0])

                i += 1
                if i == batch_size:
                    yield (lidar_d_i_s, angles)
                    i = 0
                    if False:
                        xyhist[300+(centroids[:,1]*10.).astype(np.int32), 300+(centroids[:,0]*10.).astype(np.int32)] += 1
                        seen += batch_size
                        if seen >= batch_size * (len(items) // batch_size):
                            import cv2
                            cv2.imwrite('hist.png', xyhist * 255. / np.amax(xyhist))
                            seen = 0


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


postfix = ""
metric  = "-loss{val_loss:.2f}"

save_checkpoint = ModelCheckpoint(
    "lidarnet"+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
    monitor='val_loss',
    verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=5e-7, epsilon = 0.2, cooldown = 10)

model.fit_generator(
    gen(train_items, BATCH_SIZE),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validate_items, BATCH_SIZE, training = False),
    validation_steps = len(validate_items) // BATCH_SIZE,
    epochs = 2000,
    callbacks = [save_checkpoint, reduce_lr])