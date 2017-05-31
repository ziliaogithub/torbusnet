import provider_didi
import argparse
from keras.initializers import Constant, Zeros
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model, Input
from keras.layers import Input, merge, Layer, Concatenate, Multiply, LSTM, Bidirectional
from keras.layers.merge import dot, Dot, add
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D, Cropping2D, AveragePooling2D, Conv1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam, Nadam, Adadelta, SGD
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.optimizers import Adam, Nadam, SGD
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.local import LocallyConnected1D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from torbus_layers import TorbusMaxPooling2D
import tensorflow as tf

import multi_gpu

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
parser.add_argument('-b', '--batch_size', type=int, nargs='+', default=[1], help='Batch Size during training, or list of batch sizes for each GPU, e.g. -b 12,8')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs used for training')
parser.add_argument('-d', '--dummy', action='store_true', help='Dummy data for toying')
parser.add_argument('-r', '--recurrent', action='store_true', help='Use LSTM model')

args = parser.parse_args()

MAX_EPOCH      = args.max_epoch
LEARNING_RATE  = args.learning_rate
DATA_DIR       = args.data_dir
MAX_DIST       = args.max_dist
#MAX_LIDAR_DIST = MAX_DIST + args.max_dist_offset

RINGS = range(11,21)
CHANNELS = 2
'''
11	1539
12	1890
13	1951
14	2072
15	2092
16	2171
17	2165
18	2163
19	2110

20	2151
'''
POINTS_PER_RING = 1024

assert args.gpus  == len(args.batch_size)

def get_model_recurrent():
    NRINGS = len(RINGS)
    lidar_distances   = Input(shape=(POINTS_PER_RING, NRINGS ))  # d i
    lidar_intensities = Input(shape=(POINTS_PER_RING, NRINGS ))  # d i

    l0  = Lambda(lambda x: x * 1/50. )(lidar_distances)
    l1  = Lambda(lambda x: x * 1/64. )(lidar_intensities)

    l  = Concatenate(axis=-1)([l0,l1])

    l = Bidirectional(LSTM(128, return_sequences=True, implementation=2))(l)
    l = Bidirectional(LSTM(128, return_sequences=True, implementation=2))(l)
    l = Dense(1, activation='sigmoid')(l)

    #distances = Lambda(lambda x: x * 50.)(l)
    classsification = l

    model = Model(inputs=[lidar_distances, lidar_intensities], outputs=[classsification])
    return model


if args.model:
    print("Loading model " + args.model)
    model = load_model(args.model)
    model.summary()
elif args.recurrent:
    model = get_model_recurrent()
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

if args.recurrent:
    _loss = 'binary_crossentropy'
    _metrics = ['accuracy']

model.compile(loss=_loss, optimizer='rmsprop', metrics = _metrics)

def gen(items, batch_size, points_per_ring = POINTS_PER_RING, training=True, rings = RINGS):
    #lidar_d_i_s    = np.empty((batch_size, len(rings), points_per_ring, 2))
    #lidar_rings = []
    #for i in len(rings):
    #    lidar_rings.append(np.empty((batch_size, points_per_ring, 2)))
    angles         = np.empty((batch_size, 1), dtype=np.float32)
    distances      = np.empty((batch_size, 1), dtype=np.float32)
    centroids      = np.empty((batch_size, 3), dtype=np.float32)
    boxes          = np.empty((batch_size, 8, 3), dtype=np.float32)
    lidars         = np.empty((batch_size, len(RINGS), points_per_ring, CHANNELS), dtype=np.float32)
    lidar_seqs     = np.empty((batch_size, points_per_ring, len(RINGS)), dtype=np.float32)
    intensity_seqs = np.empty((batch_size, points_per_ring, len(RINGS)), dtype=np.float32)

    # label
    classification_seqs  = np.empty((batch_size, points_per_ring,1), dtype=np.float32)

    max_angle_span = 0.
    this_max_angle = False

    lidar_rings = []
    for i in rings:
        lidar_rings.append(np.empty((batch_size, points_per_ring, CHANNELS),dtype=np.float32))

    BINS = 25.

    PAD = 0

    skip = False

    if training is True:

        # our training set is heavily unbalanced, so we are going to balance the histogram of x,y labels.
        # since we will be using rotations we will build a target histogram based on distance so that
        # the probability of a label ending up an a given x,y bucket would be roughly the same
        # we build a distance histogram target and since the number of points sampled from a distance
        # needs to grow proportionally with the radius of the circumference, we factor it in.
        _distances = []
        for item in items:
            tracklet, frame = item
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2])  # only x y
            _distances.append(distance)

        h_count,  _h_edges = np.histogram(_distances, bins=int(BINS), range=(0, MAX_DIST), density=False)
        h_edges = np.empty(_h_edges.shape[0]-1)

        for i in range(h_edges.shape[0]):
            h_edges[i] = (_h_edges[i] + _h_edges[i+1] ) / 2.

        h_edges[h_count == 0] = 0
        h_target = None
        best_min = -1
        _h_target = np.zeros_like(h_count)
        for i,r in enumerate(h_edges):
            if r > 0:
                # change this line to other target distributions of distance
                _h_target[h_count != 0] = h_count[i]
                if np.all(h_count - _h_target >= 0):
                    _min = np.amin(h_count[_h_target > 0] - _h_target[_h_target > 0])
                    if _min > best_min:
                        h_target = _h_target
                        best_min = _min
        if h_target is None:
            print("WARNING: Could not find optimal balacing set, reverting to bad one")
            h_target = np.zeros_like(h_count)
            h_target[h_count > 0] = np.amin(h_count[h_count > 0])
        h_current = h_target.copy()

        #print("Target distance histogram", h_target)

    i = 0
    seen = 0
    xyhist = np.zeros((600,600), dtype=np.float32)

    while True:
        random.shuffle(items)

        for item in items:
            tracklet, frame = item

            centroid     = tracklet.get_box_centroid(frame)[:3]
            box          = tracklet.get_box(frame).T
            distance     = np.linalg.norm(centroid[:2]) # only x y

            if training is True:

                _h = int(BINS * distance / MAX_DIST)
                if h_current[_h] == 0:
                    skip = True
                else:
                    skip = False

                    h_current[_h] -= 1
                    if np.sum(h_current) == 0:
                        h_current[:] = h_target[:]

                    # random rotation along Z axis
                    random_yaw = (np.random.random_sample() * 2. - 1.) * np.pi
                    centroid   = point_utils.rotZ(centroid, random_yaw)
                    box        = point_utils.rotZ(box, random_yaw)
                    lidar_d_i  = tracklet.get_lidar_rings(frame,
                                                         rings = rings,
                                                         points_per_ring = points_per_ring,
                                                         pad = PAD,
                                                         rotate = random_yaw,
                                                         clip = (0.,50.)) #

                    min_yaw =  np.pi
                    max_yaw = -np.pi

                    this_max_angle = False

                    for b in box[:4]:
                        yaw = np.arctan2(b[1], b[0])
                        if yaw > max_yaw:
                            max_yaw = yaw
                        if yaw < min_yaw:
                            min_yaw = yaw

                    angle_span = np.absolute(min_yaw-max_yaw)

                    if  angle_span >= np.pi:
                        skip = True
                    elif angle_span > max_angle_span:
                        max_angle_span = angle_span

                        #print("Max angle span:", max_angle_span)
                        this_max_angle = True

            else:
                lidar_d_i = tracklet.get_lidar_rings(frame,
                                                     rings = rings,
                                                     points_per_ring = points_per_ring,
                                                     pad = PAD,
                                                     clip = (0.,50.))  #

            if skip is False:

                #lidar_d_i_s[i]  = lidar_d_i#[...,0:1]
                #lidar_rings[i] = []
                for ring in rings:
                    lidar_ring     = lidar_rings[ring-rings[0]]
                    lidar_ring[i]  = lidar_d_i[ring-rings[0],:,:CHANNELS]
                    lidars[i,ring-rings[0]] = lidar_d_i[ring-rings[0],:,:CHANNELS]

                angles[i]    = np.arctan2(centroid[1], centroid[0])
                centroids[i] = centroid
                distances[i] = distance
                boxes[i]     = box

                if this_max_angle:
                    for xx in rings:
                        plt.plot(lidar_d_i[xx-rings[0],:,0])
                    xyaw = np.arctan2(centroid[1], centroid[0])
                    _yaw = int(points_per_ring * (xyaw + np.pi) / (2 * np.pi))
                    plt.axvline(x=_yaw, color='k', linestyle='--')
                    for b in box[:4]:
                        xyaw = np.arctan2(b[1], b[0])
                        _yaw = int(points_per_ring * (xyaw + np.pi) / (2 * np.pi))
                        plt.axvline(x=_yaw, color='blue', linestyle=':')

                    plt.savefig('train-max_angle.png')
                    plt.clf()
                    this_max_angle = False

                i += 1
                if i == batch_size:
                    #print(lidar_rings[0])
                    #print(lidar_rings[0][0].shape)

                    seen += batch_size
                    if seen > 1000:
                        seen = 0
                        for xx in rings:
                            plt.plot(lidar_rings[xx - rings[0]][0, :, 0])
                        yaw = np.arctan2(centroids[0][1], centroids[0][0])
                        _yaw = int(points_per_ring * (yaw + np.pi) / (2 * np.pi))
                        plt.axvline(x=_yaw, color='k', linestyle='--')
                        for b in boxes[0][:4]:
                            yaw = np.arctan2(b[1], b[0])
                            _yaw = int(points_per_ring * (yaw + np.pi) / (2 * np.pi))
                            plt.axvline(x=_yaw, color='blue', linestyle=':')

                        plt.savefig('train.png')
                        plt.clf()

                    for ii in range(batch_size):
                        for ring in rings:
                            lidar_seqs[ii,:,ring-rings[0]]     = lidars[ii,ring-rings[0],:, 0]
                            intensity_seqs[ii,:,ring-rings[0]] = lidars[ii,ring-rings[0],:, 1]

                        max_yaw = -np.pi
                        min_yaw =  np.pi
                        for b in boxes[ii][:4]:
                            yaw = np.arctan2(b[1], b[0])
                            if yaw > max_yaw:
                                max_yaw = yaw
                            if yaw < min_yaw:
                                min_yaw = yaw
                        _min_yaw = int(points_per_ring * (min_yaw + np.pi) / (2 * np.pi))
                        _max_yaw = int(points_per_ring * (max_yaw + np.pi) / (2 * np.pi))

                        classification_seqs[ii, :] = 0.
                        classification_seqs[ii, _min_yaw:_max_yaw+1] = 1. # distances[ii] for classficiation

                    yield ([lidar_seqs, intensity_seqs], classification_seqs)
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

if args.recurrent:
    postfix = "recurrent"
    metric  = "-val_acc{val_acc:.4f}"
else:
    postfix = ""
    metric  = "-val_loss{val_loss:.2f}"

save_checkpoint = ModelCheckpoint(
    "lidarnet"+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
    monitor='val_loss',
    verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=5e-7, epsilon = 0.2, cooldown = 4, verbose=1)

model.fit_generator(
    gen(train_items, BATCH_SIZE),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validate_items, BATCH_SIZE, training = False),
    validation_steps = len(validate_items) // BATCH_SIZE,
    epochs = 2000,
    callbacks = [save_checkpoint, reduce_lr])
