import provider_didi
import argparse
from keras.initializers import Constant, Zeros
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model, Input
from keras.layers import Input, merge, Layer, Concatenate, Multiply, LSTM, Bidirectional, GRU, Add
from keras.layers.merge import dot, Dot, add
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D, Cropping2D, AveragePooling2D, Conv1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam, Nadam, Adadelta, SGD
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.optimizers import Adam, Nadam, SGD, RMSprop
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.local import LocallyConnected1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.initializers import TruncatedNormal
from torbus_layers import TorbusMaxPooling2D
import tensorflow as tf
from sklearn.model_selection import train_test_split
from selu import *
import copy

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


R2_DATA_DIR      = '../didi-data/release2/Data-points-processed'
R3_DATA_DIR      = '../didi-data/release3/Data-points-processed'
R2_TRAIN_FILE    = './train-release2.txt'
R3_TRAIN_FILE    = './train-release3.txt'
R2_VALIDATE_FILE = './validate-release2.txt'
R3_VALIDATE_FILE = './validate-release3.txt'

parser = argparse.ArgumentParser()
parser.add_argument('-r2', action='store_true', help='Use release 2 car data')
parser.add_argument('-r3', action='store_true', help='Use release 3 car data')
parser.add_argument('--data-dir', default=R3_DATA_DIR, help='Tracklets top dir')
parser.add_argument('--train-file', default=R3_TRAIN_FILE, help='Fileindex for training')
parser.add_argument('--validate-file', default=R3_VALIDATE_FILE, help='Fileindex validation')
parser.add_argument('--validate-split', default=None, type=int, help='Use % percent of train set instead of fileindex, e.g --validate-split 20%' )
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run')
parser.add_argument('--max_dist', type=float, default=25, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('-b', '--batch_size', type=int, nargs='+', default=[1], help='Batch Size during training, or list of batch sizes for each GPU, e.g. -b 12,8')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs used for training')
parser.add_argument('-d', '--dummy', action='store_true', help='Dummy data for toying')
parser.add_argument('-t', '--test', action='store_true', help='Test model on validation and plot results')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-p', '--points-per-ring', action='store', type=int, default=1024, help='Number of points per lidar ring')
parser.add_argument('-r', '--rings', nargs='+', type=int, default=[10,24], help='Range of rings to use e.g. -r 10 14 uses rings 10,11,12,13 inclusive')
parser.add_argument('-u', '--unsafe-training', action='store_true', help='Use unrefined tracklets for training (UNSAFE!)')
parser.add_argument('-lo', '--localizer', action='store_true', help='Use localizer instead of classifier')
parser.add_argument('-w', '--worst-case-angle', type=float, default=1.6, help='Worst case angle in radians for localizer')
parser.add_argument('-hn', '--hidden-neurons', nargs='+', type=int, default=[64, 128, 256], help='Hidden neurons for recurrent layers, e.g. -h 64 128 256')
parser.add_argument('-fc', '--freeze-class', action='store_true', help='Freeze classification layers during training')
parser.add_argument('-fl', '--freeze-localizer', action='store_true', help='Freeze localizer layers during training')
parser.add_argument('-wl', '--weight-localizer', type=float, default=0.001, help='Weight factor for localizer component of loss')
parser.add_argument('-pn', '--pointnet', action='store_true', help='Train pointnet-based localizer')
parser.add_argument('-pp', '--pointnet-points', type=int, default=1024, help='Pointnet points for regressor')
parser.add_argument('-ss', '--sector-splits', default=4, type=int, help='Sector splits (to make it faster)')


args = parser.parse_args()

assert len(args.rings) == 2
rings = range(args.rings[0], args.rings[1])

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

MAX_EPOCH      = args.max_epoch
LEARNING_RATE  = args.learning_rate
DATA_DIR       = args.data_dir
MAX_DIST       = args.max_dist
pointnet_points = None

XML_TRACKLET_FILENAME_UNSAFE = 'tracklet_labels.xml'
XML_TRACKLET_FILENAME_SAFE   = 'tracklet_labels_trainable.xml'

UNSAFE_TRAINING = args.unsafe_training
if UNSAFE_TRAINING:
    XML_TRACKLET_FILENAME = XML_TRACKLET_FILENAME_UNSAFE
else:
    XML_TRACKLET_FILENAME = XML_TRACKLET_FILENAME_SAFE


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
assert args.gpus  == len(args.batch_size)

HUBER_DELTA = 0.5

def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    if K._BACKEND == 'tensorflow':
        import tensorflow as tf
        x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        return  K.sum(x)

def null_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

def get_model_pointnet(REGRESSION_POINTS):

    points = Input(shape=(REGRESSION_POINTS, 4))
    act = 'relu'
    ini = 'he_normal'
    p = Lambda(lambda x: x * (1. / 25., 1. / 25., 1. / 3., 1. / 64.) - (0.5, 0., -0.5, 1.))(points)
    p = Reshape(target_shape=(REGRESSION_POINTS, 4, 1), input_shape=(REGRESSION_POINTS, 4))(p)
    p = Conv2D(filters=64,   kernel_size=(1, 4), activation=act, kernel_initializer=ini)(p)
    p = Conv2D(filters=128,  kernel_size=(1, 1), dilation_rate=(1, 1), activation=act, kernel_initializer=ini)(p)
    p = Conv2D(filters=256,  kernel_size=(1, 1), dilation_rate=(1, 1), activation=act, kernel_initializer=ini)(p)
    p = Dropout(0.1)(p)
    p = Conv2D(filters=256,  kernel_size=(1, 1), dilation_rate=(1, 1), activation=act, kernel_initializer=ini)(p)
    p = Dropout(0.2)(p)
    p = Conv2D(filters=2048, kernel_size=(1, 1), dilation_rate=(1, 1), activation=act, kernel_initializer=ini)(p)
    p = Dropout(0.2)(p)

    p = MaxPooling2D(pool_size=(REGRESSION_POINTS, 1), strides=None, padding='valid')(p)

    p = Flatten()(p)
    p = Dense(512, activation=act, kernel_initializer=ini)(p)
    p = Dropout(0.1)(p)
    p = Dense(256, activation=act, kernel_initializer=ini)(p)
    p = Dropout(0.2)(p)
    c = Dense(3,   activation=None)(p)
    #s = Dense(3, activation=None)(p)

    centroids  = Lambda(lambda x: x * (12.5, 1., 3.) - (-12.5, 0., 0.8))(c)  # tx ty tz
    #dimensions = Lambda(lambda x: x * (3., 25., 25.) - (-1.5, 0., 0.))(s)  # h w l

    model = Model(inputs=points, outputs=[centroids])
    return model


def get_model_localizer(points_per_ring, rings, hidden_neurons):
    lidar_distances = Input(shape=(points_per_ring, rings))  # d i
    lidar_heights = Input(shape=(points_per_ring, rings))  # d i
    lidar_intensities = Input(shape=(points_per_ring, rings))  # d i

    l0 = Lambda(lambda x: x * 1 / 50.  - 0.5, output_shape=(points_per_ring, rings))(lidar_distances)
    l1 = Lambda(lambda x: x * 1 / 3.   + 0.5, output_shape=(points_per_ring, rings))(lidar_heights)
    l2 = Lambda(lambda x: x * 1 / 128. - 0.5, output_shape=(points_per_ring, rings))(lidar_intensities)

    l0  = Reshape((points_per_ring, rings, 1))(l0)
    l1  = Reshape((points_per_ring, rings, 1))(l1)
    l2  = Reshape((points_per_ring, rings, 1))(l2)

    l  = Concatenate(axis=-1)([l0,l1,l2])

    scales = 6
    n = 32
    nn = 64
    m = 4

    #l  = Conv2D(filters=nn//4, kernel_size=(1,1), padding='valid')(l)
    #l  = Conv2D(filters=nn//2, kernel_size=(1,1), padding='valid')(l)
    l  = Conv2D(filters=nn,    kernel_size=(1,1), padding='valid')(l)

    #rings = 1
    #mn = points_per_ring // 2 - (m // 2)
    #mm = points_per_ring // 2 + (m // 2)

    r = [ None ] * rings
    f = [ None ] * (rings * scales)
    s = [ None ] * scales
    for ring in range(rings):
        r[ring] = Lambda(lambda x: x[:, :, ring, :], output_shape=(points_per_ring, 1, nn))(l)
        r[ring] = Reshape((points_per_ring, nn))(r[ring])
        source = r[ring]
        for scale in range(scales):

            factor = int((scale+1)*points_per_ring/(15*scales)) if scale > 0 else 1
            mn = int(points_per_ring / (2*factor)) - (m // 2)
            mm = int(points_per_ring / (2*factor)) + (m // 2)
            print(factor, mn, mm)
            r[ring] = AveragePooling1D(pool_size=factor, strides=factor, padding='valid')(source) if scale > 0 else source
            print('after avgpool', r[ring])
            r[ring] = Conv1D(filters=n, kernel_size=3, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'a', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n//2, kernel_size=1, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'b', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n, kernel_size=3, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'c', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n, kernel_size=3, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'d', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n//2, kernel_size=1, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'e', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n, kernel_size=3, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'f', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n, kernel_size=3, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'g', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n//2, kernel_size=1, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'h', padding='same')(r[ring])
            r[ring] = Conv1D(filters=n, kernel_size=3, activation='relu', name='r_r'+str(ring)+'s'+str(scale)+'i', padding='same')(r[ring])

            f[ring + scale * rings] = Lambda(lambda x: x[:, mn:mm, :], output_shape=(m, n), name='f_r'+str(ring)+'s'+str(scale))(r[ring])

            if (scale == 1) and (ring ==0):
                print(f[ring + scale * rings])
                print(r[ring])

            #r[ring] = MaxPooling1D(pool_size=2, strides=2, padding='valid')(r[ring])
            if (scale == 1) and (ring ==0):
                print(r[ring])

            if ring == 0:
                print(f[ring + scale * rings])
                s[scale] = Reshape((m, 1, n))(f[ring + scale * rings])
            else:
                s[scale] = Concatenate(axis=-2, name='s_s'+str(scale) if (ring==rings-1) else None )([s[scale], Reshape((m, 1, n))(f[ring + scale * rings])])

    print(s)

    for scale in range(scales):
        if scale == 0:
            all_scales = Reshape((m , rings, 1, n))(s[scale])
        else:
            all_scales = Concatenate(axis=-2)([all_scales, Reshape((m , rings, 1, n))(s[scale])])

    print(all_scales)
    all_scales = MaxPooling3D(pool_size=(1, 1, scales))(all_scales)
    print(all_scales)

    ss = Flatten()(all_scales)
    #ss = Dense( 512, activation='relu')(ss)
    ss = Dense( 128, activation='relu')(ss)
    ss = Dense( 128, activation='relu')(ss)
    ss = Dense(   1, activation='relu')(ss)
    distances = ss

    '''
    for scale in range(scales):
        s[scale] = Flatten()(s[scale])
        s[scale] = Dense( 32, activation='relu')(s[scale])
        s[scale] = Dense( 32, activation='relu')(s[scale])
        s[scale] = Dense( 32, activation='relu')(s[scale])

    ss = s[0]
    for scale in range(scales):
        if scale > 0:
            ss = Concatenate(axis=-1)([ss, s[scale]])
    #ss = Flatten()(ss)'
    ss = Dense(32, activation='relu')(ss)
    ss = Dense(32, activation='relu')(ss)
    ss = Dense( 1, activation='relu')(ss)

    #ss = Reshape((scales, 1))(ss)
    distances = ss #GlobalMaxPooling1D()(ss)
    '''
    print(distances)

    distances = Lambda(lambda x: x * 25.)(distances)
    outputs = [distances]

    model = Model(inputs=[lidar_distances, lidar_heights, lidar_intensities], outputs=outputs)
    return model


def get_model_recurrent(points_per_ring, rings, hidden_neurons, localizer=False, sector_splits=1):
    points_per_ring = points_per_ring // sector_splits
    lidar_distances   = Input(shape=(points_per_ring, rings ))
    #lidar_heights     = Input(shape=(points_per_ring, rings ))
    lidar_intensities = Input(shape=(points_per_ring, rings ))

    l0  = Lambda(lambda x: x * 1/50.  - 0.5, output_shape=(points_per_ring, rings))(lidar_distances)
    #l1  = Lambda(lambda x: x * 1/3.   + 0.5, output_shape=(points_per_ring, rings))(lidar_heights)
    l2  = Lambda(lambda x: x * 1/128. - 0.5, output_shape=(points_per_ring, rings))(lidar_intensities)

#    o  = Concatenate(axis=-1, name='lidar')([l0, l1, l2])
    o  = Concatenate(axis=-1, name='lidar')([l0, l2])

    l  = o

    for i, hidden_neuron in enumerate(hidden_neurons):
        _return_sequences = True
        _activation = 'tanh'
        #_activation = 'relu'
        _kernel_initializer = 'he_normal'
        _name = 'class-GRU' + str(i)
        if i == (len(hidden_neurons) - 1):
            _return_sequences = True
            _activation = 'sigmoid'
            #_kernel_initializer = 'he_normal'
            _name = 'class'
        l = GRU(hidden_neuron,
                activation = _activation,
                return_sequences=_return_sequences,
                name=_name,
                kernel_initializer = _kernel_initializer,
                implementation=2)(l)
    classification = l

    outputs = [classification]


    #model = Model(inputs=[lidar_distances, lidar_heights, lidar_intensities], outputs=outputs)
    model = Model(inputs=[lidar_distances, lidar_intensities], outputs=outputs)
    return model

def save_lidar_plot(lidar_distance, box, filename, highlight=None):
    points_per_ring = lidar_distance.shape[0]
    nrings          = lidar_distance.shape[1]

    centroid = np.average(box, axis=0)

    if highlight is not None:
        assert highlight.shape[0] == points_per_ring
        for _yaw, classification in np.ndenumerate(highlight):
            if classification >= 0.5:
                plt.axvline(x=_yaw[0], alpha=0.8, color='0.8')

    for iring in range(nrings):
        plt.plot(lidar_distance[:, iring])

    yaw = np.arctan2(centroid[1], centroid[0])
    _yaw = int(points_per_ring * (yaw + np.pi) / (2 * np.pi))
    plt.axvline(x=_yaw, color='k', linestyle='--')

    for b in box[:4]:
        yaw = np.arctan2(b[1], b[0])
        _yaw = int(points_per_ring * (yaw + np.pi) / (2 * np.pi))
        plt.axvline(x=_yaw, color='blue', linestyle=':', alpha=0.5)

    plt.savefig(filename)
    plt.clf()
    return

def minmax_yaw(box):
    # initialize with worst case to make sure they get updated
    min_yaw = np.pi
    max_yaw = -np.pi

    for b in box[:4]:
        yaw = np.arctan2(b[1], b[0])
        if yaw > max_yaw:
            max_yaw = yaw
        if yaw < min_yaw:
            min_yaw = yaw
    return min_yaw, max_yaw

# if localizer_points_per_ring is a number => yield localizer semantics
def gen(items, batch_size, points_per_ring, rings, pointnet_points, sector_splits, localizer_points_per_ring=None, training=True):

    angles         = np.empty((batch_size, 1), dtype=np.float32)
    distances      = np.empty((batch_size, 1), dtype=np.float32)
    centroids      = np.empty((batch_size, 3), dtype=np.float32)
    boxes          = np.empty((batch_size, 8, 3), dtype=np.float32)

    if pointnet_points is None:
        # inputs classifier RNN
        distance_seqs  = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)
        height_seqs    = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)
        intensity_seqs = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)
        # classifier label (output): 0 if point does not belong to car, 1 otherwise
        classification_seqs = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, 1), dtype=np.float32)
    else:
    # inputs classifier pointnet
        lidars         = np.empty((batch_size, pointnet_points, 4), dtype=np.float32)

    if localizer_points_per_ring is not None:
        l_distance_seqs  = np.empty((batch_size, localizer_points_per_ring, len(rings)), dtype=np.float32)
        l_height_seqs    = np.empty((batch_size, localizer_points_per_ring, len(rings)), dtype=np.float32)
        l_intensity_seqs = np.empty((batch_size, localizer_points_per_ring, len(rings)), dtype=np.float32)

        # localizer  label (output): distance to centroid
        l_centroid_seqs  = np.empty((batch_size, 1), dtype=np.float32)

    max_angle_span = 0.
    this_max_angle = False

    BINS = 25.

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
            print("WARNING: Could not find optimal balacing set, reverting to trivial (flat) one")
            h_target = np.zeros_like(h_count)
            h_target[h_count > 0] = np.amin(h_count[h_count > 0])
        h_current = h_target.copy()

    i = 0
    seen = 0

    while True:

        if training:
            random.shuffle(items)

        for item in items:
            tracklet, frame = item

            centroid     = tracklet.get_box_centroid(frame)[:3]
            box          = tracklet.get_box(frame).T
            distance     = np.linalg.norm(centroid[:2]) # only x y

            if training:

                _h = int(BINS * distance / MAX_DIST)
                if h_current[_h] == 0:
                    skip = True
                else:
                    skip = False

                    h_current[_h] -= 1
                    if np.sum(h_current) == 0:
                        h_current[:] = h_target[:]

                    # random rotation along Z axis

                    random_yaw = (np.random.random_sample() * 2. - 1.) * np.pi if pointnet_points is None else 0.
                    centroid   = point_utils.rotZ(centroid, random_yaw)
                    box        = point_utils.rotZ(box, random_yaw)

                    flipX      = np.random.randint(2) if pointnet_points is None else 0.
                    flipY      = np.random.randint(2) if pointnet_points is None else 0.
                    if flipX:
                        centroid[0] = -centroid[0]
                        box[:,0]    = -box[:,0]
                    if flipY:
                        centroid[1] = -centroid[1]
                        box[:, 1]   = -box[:, 1]

                    min_yaw, max_yaw = minmax_yaw(box)
                    angle_span = np.absolute(min_yaw - max_yaw)
                    this_max_angle = False

                    # car is between two frames, ignore it for now
                    if angle_span >= np.pi:
                        skip = True
                    elif angle_span > max_angle_span:
                        max_angle_span = angle_span
                        # print("Max angle span:", max_angle_span)
                        # print(tracklet.xml_path, frame)
                        this_max_angle = True

                    if skip is False:
                        if pointnet_points is None:
                            lidar_d_i  = tracklet.get_lidar_rings(frame,
                                                                  rings = rings,
                                                                  points_per_ring = points_per_ring,
                                                                  clip=(0., 50.),
                                                                  rotate = random_yaw,
                                                                  flipX = flipX, flipY = flipY) #
                        else:
                            lidar = tracklet.get_lidar(frame, pointnet_points, angle_cone=(min_yaw, max_yaw), rings=rings)
                            if lidar.shape[0] == 0:
                                skip = True


            else:
                # validation
                if pointnet_points is None:
                    lidar_d_i = tracklet.get_lidar_rings(frame,
                                                         rings = rings,
                                                         points_per_ring = points_per_ring,
                                                         clip = (0.,50.))
                else:
                    min_yaw, max_yaw = minmax_yaw(box)
                    lidar = tracklet.get_lidar(frame, pointnet_points, angle_cone=(min_yaw, max_yaw), rings=rings)

            if skip is False:

                if pointnet_points is not None:
                    mean_yaw = (max_yaw + min_yaw) / 2.
                    #print('mean', mean_yaw)

                    lidar = point_utils.rotZ(lidar, mean_yaw)
                    #print(np.arctan2(centroid[1], centroid[0]))
                    centroid = point_utils.rotZ(centroid, mean_yaw)
                    #print(np.arctan2(centroid[1], centroid[0]))

                    box = point_utils.rotZ(box, mean_yaw)
                    lidars[i] = lidar[:,:4]

                angles[i]    = np.arctan2(centroid[1], centroid[0])
                #print(angles[i])
                centroids[i] = centroid
                distances[i] = distance
                boxes[i]     = box

                if pointnet_points is None:

                    min_yaw, max_yaw = minmax_yaw(box)
                    _min_yaw = int(points_per_ring * (min_yaw + np.pi) / (2 * np.pi))
                    _max_yaw = int(points_per_ring * (max_yaw + np.pi) / (2 * np.pi))
                    #print("all", _min_yaw, _max_yaw)

                    s_start = 0
                    for sector in range(sector_splits):
                        s_end = s_start + points_per_ring // sector_splits

                        classification_seqs[i * sector_splits + sector, :] = 0
                        if ((_min_yaw >= s_start) and (_min_yaw < s_end)) or \
                            ((_max_yaw >= s_start) and (_max_yaw < s_end)):
                            __min_yaw = max(s_start, _min_yaw)
                            __max_yaw = min(s_end, _max_yaw)
                            classification_seqs[i * sector_splits + sector, (__min_yaw - s_start):(__max_yaw - s_start)] = 1.
                            #print(__min_yaw,__max_yaw)

                        for ring in rings:
                            distance_seqs [i * sector_splits + sector, :, ring - rings[0]] = lidar_d_i[ring - rings[0], s_start:s_end, 0]
                            height_seqs   [i * sector_splits + sector, :, ring - rings[0]] = lidar_d_i[ring - rings[0], s_start:s_end, 1]
                            intensity_seqs[i * sector_splits + sector, :, ring - rings[0]] = lidar_d_i[ring - rings[0], s_start:s_end, 2]

                        s_start = s_end

                    if this_max_angle:
                        save_lidar_plot(distance_seqs[i], box, "train-max_angle.png")
                        this_max_angle = False

                    i += 1
                    if i == batch_size:

                        seen += batch_size
                        if seen > 1000:
                            seen = 0
                            save_lidar_plot(distance_seqs[0], boxes[0], "train.png")

                        for ii in range(batch_size):

                            min_yaw, max_yaw = minmax_yaw(boxes[ii])
                            _min_yaw = int(points_per_ring * (min_yaw + np.pi) / (2 * np.pi))
                            _max_yaw = int(points_per_ring * (max_yaw + np.pi) / (2 * np.pi))

                            if localizer_points_per_ring is None:
                                classification_seqs[ii, :] = 0.
                                classification_seqs[ii, _min_yaw:_max_yaw+1] = 1. # distances[ii] #for classification
                            else:
                                _yaw_span = _max_yaw - _min_yaw
                                l_distance_seqs[ii, :, :]  = 0.
                                l_height_seqs[ii, :, :]    = 0.
                                l_intensity_seqs[ii, :, :] = 0.

                                for ring in rings:
                                    mn = int(localizer_points_per_ring / 2. - (_yaw_span / 2.))
                                    mm = mn + _yaw_span

                                    l_distance_seqs [ii, mn:mm, ring - rings[0]] = distance_seqs [ii, _min_yaw:_max_yaw, ring - rings[0]]
                                    l_height_seqs   [ii, mn:mm, ring - rings[0]] = height_seqs   [ii, _min_yaw:_max_yaw, ring - rings[0]]
                                    l_intensity_seqs[ii, mn:mm, ring - rings[0]] = intensity_seqs[ii, _min_yaw:_max_yaw, ring - rings[0]]

                                l_centroid_seqs[ii] = np.linalg.norm(centroids[ii, :2])

                        if localizer_points_per_ring is None:
    #                        yield ([distance_seqs, intensity_seqs], [classification_seqs, distances]) #)
                            yield ([distance_seqs, intensity_seqs], [classification_seqs])
                        else:
                            yield ([l_distance_seqs, l_height_seqs, l_intensity_seqs], l_centroid_seqs) #)

                        i = 0

                else:
                    # pointnet

                    i += 1
                    if i == batch_size:
                        yield ([lidars], [centroids]) #)
                        #if training is False:
                        #    print(centroids[0])
                        i = 0


def get_items(tracklets, unsafe=False):
    items = []
    for tracklet in tracklets:
        for frame in tracklet.frames():
            state    = tracklet.get_state(frame)
            centroid = tracklet.get_box_centroid(frame)[:3]
            distance = np.linalg.norm(centroid[:2]) # only x y
            if (distance < MAX_DIST) and ( (state == 1) or unsafe):
                items.append((tracklet, frame))
    return items

def split_train(train_items, test_size):
    train_items, _validate_items = train_test_split(train_items, test_size=args.validate_split * 0.01)
    # generators are executed concurrently and Diditracklets are not thread-safe,
    # make sure no instance of tracklets are used concurrently... in other words:
    # make copies of tracklets for the validation generator
    tracklet_to_tracklet = {}
    validate_items = []
    for item in _validate_items:
        tracklet, frame = item
        if tracklet in tracklet_to_tracklet:
            _tracklet = tracklet_to_tracklet[tracklet]
        else:
            _tracklet = copy.copy(tracklet)
            tracklet_to_tracklet[tracklet] = _tracklet
        validate_items.append((_tracklet, frame))

    print("unique train_items:    " + str(len(set(train_items))))
    print("unique validate_items: " + str(len(set(validate_items))))
    return train_items, validate_items

if args.model:
    print("Loading model " + args.model)

    # monkey-patch null_loss so model loads ok
    # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
    import keras.losses
    keras.losses.null_loss = null_loss

    model = load_model(args.model)
    points_per_ring = model.get_input_shape_at(0)[0][1] * args.sector_splits
    nrings = model.get_input_shape_at(0)[0][2]
    print('Loaded model with ' + str(points_per_ring) + ' points per ring and ' + str(nrings) + ' rings')
    #assert points_per_ring == args.points_per_ring
    assert nrings == len(rings)
    localizer_points_per_ring = int(np.ceil(points_per_ring * args.worst_case_angle / (2 * np.pi ))) if args.localizer else None

else:
    points_per_ring = args.points_per_ring
    localizer_points_per_ring = int(np.ceil(points_per_ring * args.worst_case_angle / (2 * np.pi ))) if args.localizer else None
    if args.pointnet:
        pointnet_points = args.pointnet_points
        model = get_model_pointnet(pointnet_points)
    else:
        if args.localizer:
            model = get_model_localizer(localizer_points_per_ring, len(rings), hidden_neurons=args.hidden_neurons)
        else:
            model = get_model_recurrent(
                localizer_points_per_ring if args.localizer else points_per_ring,
                len(rings),
                hidden_neurons=args.hidden_neurons,
                localizer=args.localizer,
                sector_splits = args.sector_splits)

if (args.gpus > 1) or (len(args.batch_size) > 1):
    if len(args.batch_size) == 1:
        model = multi_gpu.make_parallel(model, args.gpus )
        BATCH_SIZE = args.gpus * args.batch_size[0]
    else:
        BATCH_SIZE = sum(args.batch_size)
        model = multi_gpu.make_parallel(model, args.gpus , splits=args.batch_size)
else:
    BATCH_SIZE = args.batch_size[0]

if args.validate_split is None:
    if args.r2 or args.r3:
        validate_items = [ ]
        if args.r2:
            validate_items.extend(get_items(provider_didi.get_tracklets(R2_DATA_DIR, R2_VALIDATE_FILE, xml_filename=XML_TRACKLET_FILENAME_SAFE), unsafe=False))
        if args.r3:
            validate_items.extend(get_items(provider_didi.get_tracklets(R3_DATA_DIR, R3_VALIDATE_FILE, xml_filename=XML_TRACKLET_FILENAME_UNSAFE), unsafe=True))
    else:
        validate_items = get_items(provider_didi.get_tracklets(DATA_DIR, args.validate_file, xml_filename=XML_TRACKLET_FILENAME), unsafe=UNSAFE_TRAINING)

if args.localizer or args.pointnet:
    _loss = 'mse'
    _metrics = ['mse']
else:
    _class_loss = 'binary_crossentropy' if not args.freeze_class     else null_loss
    _loc_loss   = 'mse'                 if not args.freeze_localizer else null_loss
    _loss = 'binary_crossentropy'#'[_class_loss, _loc_loss]
    _metrics = { 'class':'acc'}


model.summary()
#model.compile(loss=_loss, optimizer=RMSprop(lr=LEARNING_RATE), metrics = _metrics, loss_weights = [ 1., args.weight_localizer] if not args.localizer and not args.pointnet else [1.])
model.compile(loss=_loss, optimizer=RMSprop(lr=LEARNING_RATE), metrics = _metrics)

if args.test:
    distance_seq = np.empty((points_per_ring, len(rings)), dtype=np.float32)
    if args.validate_split is not None:

        _items = get_items(provider_didi.get_tracklets(DATA_DIR, args.train_file, xml_filename=XML_TRACKLET_FILENAME), unsafe=UNSAFE_TRAINING)
        _, validate_items = split_train(_items, test_size=args.validate_split * 0.01)

    _predictions = model.predict_generator(
        #        gen(train_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, localizer_points_per_ring),

        generator=gen(validate_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, args.sector_splits, localizer_points_per_ring, training = False),
        steps=args.sector_splits * len(validate_items) // BATCH_SIZE)

    print(_predictions.shape)

    predictions =  _predictions.reshape((-1, points_per_ring, 1))
    print(predictions.shape)
    i = 0
    for item,prediction in zip(validate_items,predictions):
        tracklet, frame = item
        lidar_d_i = tracklet.get_lidar_rings(frame,
                                             rings=rings,
                                             points_per_ring=points_per_ring,
                                             clip=(0., 50.))
        box = tracklet.get_box(frame).T

        for ring in rings:
            distance_seq[:, ring - rings[0]] = lidar_d_i[ring - rings[0], :, 0]

        save_lidar_plot(distance_seq, box, os.path.join('test', str(i) + '.png'), highlight = prediction)
        i += 1

else:
    if args.r2 or args.r3:
        train_items = [ ]
        if args.r2:
            train_items.extend(get_items(provider_didi.get_tracklets(R2_DATA_DIR, R2_TRAIN_FILE, xml_filename=XML_TRACKLET_FILENAME_SAFE), unsafe=False))
        if args.r3:
            train_items.extend(get_items(provider_didi.get_tracklets(R3_DATA_DIR, R3_TRAIN_FILE, xml_filename=XML_TRACKLET_FILENAME_UNSAFE), unsafe=True))
    else:
        train_items = get_items(provider_didi.get_tracklets(DATA_DIR, args.train_file, xml_filename=XML_TRACKLET_FILENAME), unsafe=UNSAFE_TRAINING)
    if args.validate_split is not None:
        train_items, validate_items = split_train(train_items, test_size=args.validate_split * 0.01)

    print("Train items:    " + str(len(train_items)))
    print("Validate items: " + str(len(validate_items)))

    if args.localizer or args.pointnet:
        postfix = "-loc-rings_"+str(rings[0])+'_'+str(rings[-1]+1)
        metric  = "-val_loss{val_loss:.4f}"
        monitor = 'val_loss'
    else:
        postfix = "-cla-rings_"+str(rings[0])+'_'+str(rings[-1]+1)+'-sectors_'+str(args.sector_splits)
        metric  = "-val_class_acc{val_acc:.4f}"
        monitor = 'val_acc'


    save_checkpoint = ModelCheckpoint(
        "lidarnet"+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
        monitor=monitor,
        verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-7, epsilon = 0.0001, verbose=1)
    print(len(validate_items) // BATCH_SIZE)
    model.fit_generator(
        generator        = gen(train_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, args.sector_splits, localizer_points_per_ring),
        steps_per_epoch  = len(train_items) // BATCH_SIZE,
        validation_data  = gen(validate_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, args.sector_splits, localizer_points_per_ring, training = False),
        validation_steps = len(validate_items) // BATCH_SIZE,
        epochs = args.max_epoch,
        callbacks = [save_checkpoint, reduce_lr])
