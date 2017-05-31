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

def get_dummy_model():
    # receptive field needs to be 1/5 of total to detect large objects close by (e.g. for 1024 -> 205)

    lidar_signal = Input(shape=(POINTS_PER_RING,1))  # d i

    d = Lambda(lambda x: x * (1 / 50.) - (0.5))(lidar_signal)
    
    act  = 'relu'
    actb = 'elu'

    for i in range(16): # block as 10 px receptive field
        c = Conv1D(filters=  64, kernel_size= 5, activation=act, padding='same')(d)
        c = Conv1D(filters=  32, kernel_size= 3, activation=act, padding='same')(c)
        c = Conv1D(filters=  64, kernel_size= 5, activation=act, padding='same')(c)
        d = add([c,d])

    d = Conv1D(filters=  32, kernel_size=1, activation=act, padding='same')(d)#, kernel_regularizer=l2(0.1), activity_regularizer=l2(0.1))(d)

    f = Flatten()(d)
    d = Dense( 32, activation=actb)(f)
    d = Dense( 16, activation=actb)(d)
    d = Dense(  8, activation=actb)(d)
    d = Dense(1)(d)

    a = Dense( 32, activation=actb)(f)
    a = Dense( 16, activation=actb)(a)
    a = Dense(  8, activation=actb)(a)
    a = Dense(1)(a)
    distance = Lambda(lambda x: x * (50.) + (25.))(d)
    angle    = Lambda(lambda x: x  + 0.5) (a)

    model = Model(inputs=lidar_signal, outputs=[distance, angle])
    return model


def angle_loss(angle_true, angle_pred):
    print(angle_true.shape)
    print(angle_pred.shape)

    xx_true = K.cos(angle_true)
    yy_true = K.sin(angle_true)

    xx_pred = K.cos(angle_pred)
    yy_pred = K.sin(angle_pred)
    vector_true = K.concatenate([xx_true, yy_true], axis=-1)
    vector_pred = K.concatenate([xx_pred, yy_pred], axis=-1)
    print('vector_true', vector_true.shape)
    print('vector_pred', vector_pred.shape)

    return K.mean(K.square(vector_true - vector_pred), axis=-1)

def get_model_recurrent():
    NRINGS = len(RINGS)
    lidar_distances   = Input(shape=(POINTS_PER_RING, NRINGS ))  # d i
    lidar_intensities = Input(shape=(POINTS_PER_RING, NRINGS ))  # d i

    l0  = Lambda(lambda x: x * 1/50. )(lidar_distances)
    l1  = Lambda(lambda x: x * 1/64. )(lidar_intensities)

    l  = Concatenate(axis=-1)([l0,l1])

    l = Bidirectional(LSTM(64, return_sequences=True, implementation=2))(l)
    l = Bidirectional(LSTM(64, return_sequences=True, implementation=2))(l)
    l = Dense(1, activation='sigmoid')(l)

    #distances = Lambda(lambda x: x * 50.)(l)
    classsification = l

    model = Model(inputs=[lidar_distances, lidar_intensities], outputs=[classsification])
    return model


def get_model():
    # receptive field needs to be 1/5 of total to detect large objects close by (e.g. for 1024 -> 205)
    NRINGS = len(RINGS)
    #ring_points = Input(shape=(RINGS, POINTS_PER_RING, 3)) # x y z

    p0  =  r0 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p1  =  r1 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p2  =  r2 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p3  =  r3 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p4  =  r4 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p5  =  r5 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p6  =  r6 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p7  =  r7 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p8  =  r8 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i
    p9  =  r9 = Input(shape=(POINTS_PER_RING, CHANNELS)) # d i



    # assume d  90 max =>  90 -> 1, 0 -> -1
    #        i 128 max => 128 -> 1, 0 -> -1

    if CHANNELS == 2:
        p0 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r0)
        p1 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r1)
        p2 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r2)
        p3 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r3)
        p4 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r4)
        p5 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r5)
        p6 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r6)
        p7 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r7)
        p8 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r8)
        p9 =  Lambda(lambda x: x * (1/50. , 1. / 64.) - (0.5, 1.))(r9)

    else:
        p0  = Lambda(lambda x: x * 1/50. - 0.5)(r0)
        p1  = Lambda(lambda x: x * 1/50. - 0.5)(r1)
        p2  = Lambda(lambda x: x * 1/50. - 0.5)(r2)
        p3  = Lambda(lambda x: x * 1/50. - 0.5)(r3)
        p4  = Lambda(lambda x: x * 1/50. - 0.5)(r4)
        p5  = Lambda(lambda x: x * 1/50. - 0.5)(r5)
        p6  = Lambda(lambda x: x * 1/50. - 0.5)(r6)
        p7  = Lambda(lambda x: x * 1/50. - 0.5)(r7)
        p8  = Lambda(lambda x: x * 1/50. - 0.5)(r8)
        p9  = Lambda(lambda x: x * 1/50. - 0.5)(r9)



    filter = 64
    p0  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p0)
    p1  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p1)
    p2  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p2)
    p3  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p3)
    p4  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p4)
    p5  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p5)
    p6  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p6)
    p7  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p7)
    p8  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p8)
    p9  = Conv1D(filters=filter,    kernel_size=3, padding='same', activation='relu')(p9)

    k1 = 5
    k2 = 3
    for i, filters in enumerate([64] * 20): # each block as 10px receptive field
        p0r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p0)
        p0r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p0r)
        p0  = add([p0,p0r])

        p1r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p1)
        p1r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p1r)
        p1  = add([p1,p1r])

        p2r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p2)
        p2r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p2r)
        p2  = add([p2,p2r])

        p3r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p3)
        p3r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p3r)
        p3  = add([p3,p3r])

        p4r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p4)
        p4r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p4r)
        p4  = add([p4,p4r])

        p5r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p5)
        p5r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p5r)
        p5  = add([p5,p5r])

        p6r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p6)
        p6r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p6r)
        p6  = add([p6,p6r])

        p7r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p7)
        p7r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p7r)
        p7  = add([p7,p7r])

        p8r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p8)
        p8r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p8r)
        p8  = add([p8,p8r])

        p9r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p9)
        p9r = Conv1D(filters=filters,   kernel_size=k1, padding='same', activation='relu')(p9r)
        p9  = add([p9,p9r])

        if i in [4,10]:
            p0 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p0)
            p1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p1)
            p2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p2)
            p3 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p3)
            p4 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p4)
            p5 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p5)
            p6 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p6)
            p7 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p7)
            p8 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p8)
            p9 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p9)

    if False:
        p0 = Lambda(nuke)(p0)
        p1 = Lambda(nuke)(p1)
        p2 = Lambda(nuke)(p2)
        p3 = Lambda(nuke)(p3)
        p4 = Lambda(nuke)(p4)
        #p5 = Lambda(nuke)(p5)
        #p6 = Lambda(nuke)(p6)
        #p7 = Lambda(nuke)(p7)
        #p8 = Lambda(nuke)(p8)
        p9 = Lambda(nuke)(p9)
        p10 = Lambda(nuke)(p10)
        p11 = Lambda(nuke)(p11)

    def nuke(x):
        zeros = tf.zeros((POINTS_PER_RING, 1))
        return tf.multiply(x, zeros)

    p = Concatenate(axis=2)([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9])

    #p = MaxPooling1D(pool_size=2, strides=2, padding='valid')(p)

    p  = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(p)
    pr = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(p)
    p  = add([p, pr])
    pr = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(p)
    p  = add([p, pr])
    pr = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(p)
    p  = add([p, pr])
    p  = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(p)

    def mul_angle_axis(x):
        angle_axis = tf.linspace(-np.pi, (POINTS_PER_RING - 1) * np.pi / POINTS_PER_RING, num=POINTS_PER_RING)
        angle_axis = tf.expand_dims(angle_axis, axis=-1)
        return tf.multiply(x, angle_axis)

    #p = Lambda(mul_angle_axis)(p)
    #angle = MaxPooling1D(pool_size=POINTS_PER_RING)(p)

    p = Flatten()(p)
    p = Dense( 64, activation='elu')(p)
    p = Dense( 64, activation='elu')(p)
    p = Dense( 32, activation='elu')(p)
    p = Dense( 16, activation='elu')(p)

    angle = Dense(1)(p)

    model = Model(inputs=[r0,r1,r2,r3,r4,r5,r6,r7,r8,r9], outputs=[angle])
    return model

if args.dummy:
    model = get_dummy_model()
    model.summary()
elif args.model:
    print("Loading model " + args.model)
    model = load_model(args.model)
    model.summary()
elif args.recurrent:
    model = get_model_recurrent()
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

if args.dummy:
    _loss = 'mse'
    _metrics = []
elif args.recurrent:
    _loss = 'binary_crossentropy'
    _metrics = ['accuracy']
else:
    _loss = angle_loss
    _metrics = []

model.compile(loss=_loss, optimizer='rmsprop', metrics = _metrics)

def gen_dummy(batch_size, points_per_ring = POINTS_PER_RING, training=True, rings = RINGS):

    lidar_signals  = np.empty((batch_size, points_per_ring,1))
    angles         = np.empty((batch_size,1))
    distances      = np.empty((batch_size,1))

    i = 0
    while True:

        distance     = 25. + 20 * (np.random.rand() - 0.5)

        angle        = ((np.random.rand() - 0.5) * 0.35 + 0.5 )* points_per_ring # np.pi/4.
        x_axis       = np.linspace(0.,points_per_ring-1,num=points_per_ring)
        width = 300. + 200 * (np.random.rand() - 0.5)
        lidar_signal = 50. - distance * (np.exp((-(x_axis - angle)**2)/width))

        lidar_signals[i,:] = np.expand_dims(lidar_signal,axis=-1)
        distances[i,:]     = distance
        angles[i,:]        = angle/points_per_ring

        i += 1

        if i == batch_size:
            yield(lidar_signals, [distances, angles])
            i = 0



def gen(items, batch_size, points_per_ring = POINTS_PER_RING, training=True, rings = RINGS, recurrent=False):
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

                    if recurrent:
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
                    else:
                        yield (lidar_rings, angles)
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

if  args.dummy:
    model.fit_generator(
        gen_dummy(BATCH_SIZE),
        steps_per_epoch=20000 // BATCH_SIZE,
        validation_data=gen_dummy(BATCH_SIZE, training=False),
        validation_steps=4000 // BATCH_SIZE,
        epochs=200)
else:
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
        gen(train_items, BATCH_SIZE, recurrent=args.recurrent),
        steps_per_epoch  = len(train_items) // BATCH_SIZE,
        validation_data  = gen(validate_items, BATCH_SIZE, recurrent=args.recurrent, training = False),
        validation_steps = len(validate_items) // BATCH_SIZE,
        epochs = 2000,
        callbacks = [save_checkpoint, reduce_lr])
