import provider_didi
import argparse
from keras.models import Model
from keras.layers import Input, Concatenate, GRU, Bidirectional
from keras.layers.core import Dense, Flatten, Lambda, Dropout, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import copy
from keras import backend as K

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

from diditracklet import *
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
parser.add_argument('-ms', '--model-suffix', default=None, type=str, help='Save model with provided suffix e.g. --model-suffix car')
parser.add_argument('--data-dir', default=R3_DATA_DIR, help='Tracklets top dir')
parser.add_argument('--train-file', default=R3_TRAIN_FILE, help='Fileindex for training')
parser.add_argument('--validate-file', default=R3_VALIDATE_FILE, help='Fileindex validation')
parser.add_argument('--validate-split', default=None, type=int, help='Use percent of train set instead of fileindex, e.g for 20% --validate-split 20' )
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run')
parser.add_argument('--max_dist', type=float, default=25, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('-b', '--batch_size', type=int, nargs='+', default=[1], help='Batch Size during training, or list of batch sizes for each GPU, e.g. -b 12,8')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs used for training')
parser.add_argument('-t', '--test', action='store_true', help='Test model on validation and plot results')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-p', '--points-per-ring', action='store', type=int, default=1024, help='Number of points per lidar ring')
parser.add_argument('-r', '--rings', nargs='+', type=int, default=[10,24], help='Range of rings to use e.g. -r 10 14 uses rings 10,11,12,13 inclusive')
parser.add_argument('-u', '--unsafe-training', action='store_true', help='Use unrefined tracklets for training (UNSAFE!)')
parser.add_argument('-hn', '--hidden-neurons', nargs='+', type=int, default=[64, 128, 256], help='Hidden neurons for recurrent layers, e.g. -h 64 128 256')
parser.add_argument('-pn', '--pointnet', action='store_true', help='Train pointnet-based localizer')
parser.add_argument('-pp', '--pointnet-points', type=int, default=1024, help='Pointnet points for regressor')
parser.add_argument('-ss', '--sector-splits', default=16, type=int, help='Sector splits (to make it faster)')
parser.add_argument('-sw', '--scale-w', default=1., type=float, action='store', help='Scale bounding box width ')
parser.add_argument('-sl', '--scale-l', default=1., type=float, action='store', help='Scale bounding box width ')
parser.add_argument('-sh', '--scale-h', default=1., type=float, action='store', help='Scale bounding box width ')
parser.add_argument('-bd', '--bidirectional-first-pass', action='store_true', help='Make first layer of RNN bidirectional')
parser.add_argument('-nc', '--normalize-car', action='store_true', help='Normalize car segmenter using pre-computed (hardcoded!) mean/var')
parser.add_argument('-np', '--normalize-ped', action='store_true', help='Normalize pedestrian segmenter using pre-computed (hardcoded!) mean/var')


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
BOX_SCALING    = (args.scale_h, args.scale_w, args.scale_l)
CLIP_DIST      = (0., 50.)
CLIP_HEIGHT    = (-3., 1.)

NORMALIZE_CAR = args.normalize_car
NORMALIZE_PED = args.normalize_ped


XML_TRACKLET_FILENAME_UNSAFE = 'tracklet_labels.xml'
XML_TRACKLET_FILENAME_SAFE   = 'tracklet_labels_trainable.xml'

UNSAFE_TRAINING = args.unsafe_training
if UNSAFE_TRAINING:
    XML_TRACKLET_FILENAME = XML_TRACKLET_FILENAME_UNSAFE
else:
    XML_TRACKLET_FILENAME = XML_TRACKLET_FILENAME_SAFE

assert args.gpus  == len(args.batch_size)

def min_angle_diff(y_true, y_pred):
    if K._BACKEND == 'theano':
        import theano
        arctan2 = theano.tensor.arctan2
    elif K._BACKEND == 'tensorflow': # NOT TESTED
        import tensorflow as tf
        arctan2 = tf.atan2

    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    vector_diff = K.abs(arctan2(K.sin(y_true-y_pred), K.cos(y_true-y_pred)))
    return K.mean(vector_diff, axis=-1)

def min_angle_error_degrees(y_true, y_pred):
    return min_angle_diff(y_true, y_pred) * 180. / np.pi

def angle_loss(y_true, y_pred):
    if K._BACKEND == 'theano':
        import theano
        arctan2 = theano.tensor.arctan2
    elif K._BACKEND == 'tensorflow': # https://www.tensorflow.org/api_docs/python/tf/atan2
        import tensorflow as tf
        arctan2 = tf.atan2

    vector_diff_square = K.square(K.cos(y_true) - K.cos(y_pred)) + K.square(K.sin(y_true) - K.sin(y_pred))
    return K.mean(vector_diff_square, axis=-1)

def get_model_pointnet(REGRESSION_POINTS):

    CHANNELS = 4

    points   = Input(shape=(REGRESSION_POINTS, CHANNELS))
    distance = Input(shape=(1,))

    act = 'relu'
    ini = 'glorot_uniform'
    p = Reshape(target_shape=(REGRESSION_POINTS, CHANNELS, 1), input_shape=(REGRESSION_POINTS, CHANNELS))(points)
    p = Conv2D(filters=64,   kernel_size=(1, CHANNELS), activation=act, kernel_initializer=ini)(p)
    p = Conv2D(filters=128,  kernel_size=(1, 1), activation=act, kernel_initializer=ini)(p)
    p = Conv2D(filters=256,  kernel_size=(1, 1), activation=act, kernel_initializer=ini)(p)
    p = Conv2D(filters=2048, kernel_size=(1, 1), activation=act, kernel_initializer=ini)(p)

    p = MaxPooling2D(pool_size=(REGRESSION_POINTS, 1), padding='valid')(p)

    p = Flatten()(p)
    p = Dense(512, activation=act)(p)
    p = Dropout(0.1)(p)
    pyaw = psize = p
    p = Dense(256, activation=act)(p)
    p = Dropout(0.2)(p)
    centroids  = Dense(3, name='centroid')(p)

    psize = Dense(64, activation=act)(psize)
    psize = Dropout(0.2)(psize)
    box_sizes = Dense(3, name='box_size')(psize)

    pyaw = Concatenate()([pyaw, distance, centroids])
    pyaw = Dense(256, activation=act)(pyaw)
    pyaw = Dense(128, activation=act)(pyaw)
    pyaw = Dense(64,  activation=act)(pyaw)
    pyaw = Dropout(0.1)(pyaw)
    pyaw = Dense(32,  activation=act)(pyaw)
    yaws = Dense(1,   activation='tanh')(pyaw)
    yaws = Lambda(lambda x: x * np.pi / 2., name='yaw', output_shape=(1,))(yaws)

    model = Model(inputs=[points, distance], outputs=[centroids, box_sizes, yaws])

    return model

def get_model_recurrent(points_per_ring, rings, hidden_neurons, sector_splits=1, bidirectional_first_pass=False):
    points_per_ring = points_per_ring // sector_splits
    lidar_distances   = Input(shape=(points_per_ring, rings ))
    lidar_heights     = Input(shape=(points_per_ring, rings ))
    lidar_intensities = Input(shape=(points_per_ring, rings ))

    if NORMALIZE_CAR:
        # These values make sure mean = 0 var = 1.
        # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        l0  = Lambda(lambda x: x * 1/17.88  - 1.35,   output_shape=(points_per_ring, rings), name='car_d_mean0_var1_norm')(lidar_distances)
        l1  = Lambda(lambda x: x * 1/1.14   + 0.675,  output_shape=(points_per_ring, rings), name='car_h_mean0_var1_norm')(lidar_heights)
        l2  = Lambda(lambda x: x * 1/20.    - 0.97,   output_shape=(points_per_ring, rings), name='car_i_mean0_var1_norm')(lidar_intensities)
    elif NORMALIZE_PED:
        l0  = Lambda(lambda x: x * 1/11.6   - 1.019,  output_shape=(points_per_ring, rings), name='ped_d_mean0_var1_norm')(lidar_distances)
        l1  = Lambda(lambda x: x * 1/0.8481 + 0.3977, output_shape=(points_per_ring, rings), name='ped_h_mean0_var1_norm')(lidar_heights)
        l2  = Lambda(lambda x: x * 1/64.35  - 3.3132, output_shape=(points_per_ring, rings), name='ped_i_mean0_var1_norm')(lidar_intensities)
    else:
        l0  = Lambda(lambda x: x * 1/50.    - 0.5,    output_shape=(points_per_ring, rings), name='avg_d_norm')(lidar_distances)
        l1  = Lambda(lambda x: x * 1/4.     + 0.25,   output_shape=(points_per_ring, rings), name='avg_h_norm')(lidar_heights)
        l2  = Lambda(lambda x: x * 1/255.   - 0.5,    output_shape=(points_per_ring, rings), name='avg_i_norm')(lidar_intensities)

    o  = Concatenate(axis=-1, name='lidar')([l0, l1, l2])

    l  = o

    for i, hidden_neuron in enumerate(hidden_neurons):
        _return_sequences = True
        _activation = 'tanh'
        #_activation = 'relu'
        _kernel_initializer = 'he_normal'
        _name = 'class-GRU' + str(i)
        _dropout = 0.1
        if i == (len(hidden_neurons) - 1):
            _return_sequences = True
            _activation = 'sigmoid'
            #_kernel_initializer = 'he_normal'
            _name = 'class'
            hidden_neuron = rings
            _dropout = 0.2
        if bidirectional_first_pass and i ==0:
            l = Bidirectional(GRU(hidden_neuron,
                    activation = _activation,
                    return_sequences=_return_sequences,
                    name=_name,
                    dropout = _dropout,
                    kernel_initializer = _kernel_initializer,
                    implementation=2))(l)
        else:
            l = GRU(hidden_neuron,
                    activation = _activation,
                    return_sequences=_return_sequences,
                    name=_name,
                    dropout = _dropout,
                    kernel_initializer = _kernel_initializer,
                    implementation=2)(l)
    classification = l

    outputs = [classification]


    model = Model(inputs=[lidar_distances, lidar_heights, lidar_intensities], outputs=outputs)
    return model

def save_lidar_plot(lidar_distance, box, filename, highlight=None, lidar_distance_gt=None, lidar_distance_pred=None):
    points_per_ring = lidar_distance.shape[0]
    nrings          = lidar_distance.shape[1]

    fig, subplots = plt.subplots(nrows=1, ncols=1 if lidar_distance_pred is None else 2, sharex=True, squeeze=False)

    if highlight is not None:
        assert highlight.shape[0] == points_per_ring
        for _yaw, classification in np.ndenumerate(highlight):
            if classification >= 0.5:
                subplots[0].axvline(x=_yaw[0], alpha=0.8, color='0.8')

    black  = np.array([0.,0.,0.,0.8]) * np.ones((points_per_ring,4))
    angles = np.linspace(0, (points_per_ring - 1), num=points_per_ring)
    point_size = 0.5
    for iring in range(nrings):
        if lidar_distance_gt is None:
            subplots[0,0].plot(lidar_distance[:, iring])
        else:
            ring_lidar_distance_gt = lidar_distance_gt[iring]
            colors = np.array(black)
            colors[:, 0] +=  ring_lidar_distance_gt
            subplots[0,0].scatter(angles, lidar_distance[:, iring], s=point_size, color=colors)
            if lidar_distance_pred is not None:
                ring_lidar_distance_pred = lidar_distance_pred[:,iring]
                colors = np.array(black)
                colors[ring_lidar_distance_pred >= 0.5, 0] = 1.
                subplots[0, 1].scatter(angles, lidar_distance[:, iring], s=point_size, color=colors)

    if box is not None:
        centroid = np.average(box, axis=0)

        yaw = np.arctan2(centroid[1], centroid[0])
        _yaw = int(points_per_ring * (yaw + np.pi) / (2 * np.pi))
        subplots[0,0].axvline(x=_yaw, color='k', linestyle='--')

        for b in box[:4]:
            yaw = np.arctan2(b[1], b[0])
            _yaw = int(points_per_ring * (yaw + np.pi) / (2 * np.pi))
            subplots[0,0].axvline(x=_yaw, color='blue', linestyle=':', alpha=0.5)

    fig.savefig(filename)
    fig.clf()
    plt.close(fig)
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

def gen(items, batch_size, points_per_ring, rings, pointnet_points, sector_splits, training=True):

    centroids      = np.empty((batch_size, 3), dtype=np.float32)

    if pointnet_points is None:
        # inputs classifier RNN
        distance_seqs  = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)
        height_seqs    = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)
        intensity_seqs = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)
        # classifier label (output): 0 if point does not belong to car, 1 otherwise
        classification_seqs      = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, 1), dtype=np.float32)

    else:
        # inputs classifier pointnet
        lidars         = np.empty((batch_size, pointnet_points, 4), dtype=np.float32)
        distances      = np.empty((batch_size, 1), dtype=np.float32)

        # extra outputs
        box_sizes      = np.empty((batch_size, 3), dtype=np.float32)
        yaws           = np.empty((batch_size, 1), dtype=np.float32)

    ring_classification_seqs = np.empty((batch_size * sector_splits, points_per_ring // sector_splits, len(rings)), dtype=np.float32)

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


    yielded = 0
    distance_cumsum  = height_cumsum  = intensity_cumsum  = 0.
    distance_cumdev2 = height_cumdev2 = intensity_cumdev2 = 0.

    distance_mean = height_mean = intensity_mean = None
    print_mean = print_var = True

    while True:

        if training:
            random.shuffle(items)

        for item in items:
            tracklet, frame = item

            centroid     = tracklet.get_box_centroid(frame)[:3]
            box          = tracklet.get_box(frame).T
            distance     = np.linalg.norm(centroid[:2]) # only x y

            if training:

                # balance training set by distance by skipping samples if we've already seen enough
                # from the same distance range.
                _h = int(BINS * distance / MAX_DIST)
                if h_current[_h] == 0:
                    skip = True
                else:
                    skip = False

                    h_current[_h] -= 1
                    if np.sum(h_current) == 0:
                        h_current[:] = h_target[:]

                    # random rotation along Z axis (doesn' make any sense in localizer)
                    random_yaw = (np.random.random_sample() * 2. - 1.) * np.pi if pointnet_points is None else 0.
                    centroid   = point_utils.rotZ(centroid, random_yaw)
                    box        = point_utils.rotZ(box, random_yaw)

                    # TODO: In pointnet we could flip Y
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

                    # car is between two frames, ignore it for now.
                    # TODO: Calculate if car is between two frames
                    if False: #angle_span >= np.pi:
                        skip = True
                    elif angle_span > max_angle_span:
                        max_angle_span = angle_span
                        # print("Max angle span:", max_angle_span)
                        # print(tracklet.xml_path, frame)
                        this_max_angle = True

                    if skip is False:
                        lidar_d_i, lidar_int  = tracklet.get_lidar_rings(frame,
                                                              rings = rings,
                                                              points_per_ring = points_per_ring,
                                                              clip = CLIP_DIST,
                                                              clip_h = CLIP_HEIGHT,
                                                              rotate = random_yaw,
                                                              flipX = flipX, flipY = flipY,
                                                              return_lidar_interpolated = True)


            else:
                # validation
                lidar_d_i, lidar_int = tracklet.get_lidar_rings(frame,
                                                     rings = rings,
                                                     points_per_ring = points_per_ring,
                                                     clip = CLIP_DIST,
                                                     clip_h = CLIP_HEIGHT,
                                                     return_lidar_interpolated=True)

            if skip is False:

                # pointnet
                if pointnet_points is not None:
                    points_in_box = DidiTracklet.get_lidar_in_box(lidar_int, box.T)
                    if points_in_box.shape[0] == 0:
                        print(points_in_box.shape)
                        print(tracklet.xml_path, frame)
                        if args.unsafe_training:
                            skip = True
                            break
                        else:
                            assert False

                    points_in_box = DidiTracklet.resample_lidar(points_in_box, pointnet_points)

                    yaw_correction = 0.
                    if True:
                        points_in_box_mean = np.mean(points_in_box[:, :3], axis=0)
                        angle = np.arctan2(points_in_box_mean[1], points_in_box_mean[0])

                        centroid = point_utils.rotZ(centroid, angle)
                        box = point_utils.rotZ(box, angle)
                        points_in_box = point_utils.rotZ(points_in_box, angle)
                        yaw_correction = -angle

                    points_in_box_mean = np.mean(points_in_box[:, :3], axis=0)
                    points_in_box[:, :3] -= points_in_box_mean
                    centroid -= points_in_box_mean
                    lidars[i] = points_in_box[:,:4]
                    lidars[i,:,3] /= 128.

                    distances[i] = np.linalg.norm(points_in_box_mean[:2])
                    box_sizes[i] = tracklet.get_box_size()

                    yaw     = point_utils.remove_orientation(tracklet.get_yaw(frame))
                    yaws[i] = point_utils.remove_orientation(yaw + yaw_correction)

                centroids[i] = centroid

                if pointnet_points is None:

                    min_yaw, max_yaw = minmax_yaw(box)

                    _min_yaw = int(points_per_ring * (min_yaw + np.pi) / (2 * np.pi))
                    _max_yaw = int(points_per_ring * (max_yaw + np.pi) / (2 * np.pi))

                    point_idx_in_box = DidiTracklet.get_lidar_in_box(lidar_int, box.T, return_idx_only=True)

                    ring_classification = np.zeros((len(rings), points_per_ring ), dtype=np.float32)
                    ring_classification[np.floor_divide(point_idx_in_box, points_per_ring), np.remainder(point_idx_in_box, points_per_ring)] = 1.

                    #print(ring_classification[:,4])

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

                            ring_classification_seqs[i * sector_splits + sector, :, ring - rings[0]] = ring_classification[ring - rings[0], s_start:s_end]

                        s_start = s_end

                    if this_max_angle:
                        save_lidar_plot(distance_seqs[i], box, "train-max_angle.png")
                        this_max_angle = False

                i += 1

                if i == batch_size:
                    yielded += batch_size * sector_splits

                    if pointnet_points is None:
                        yield ([distance_seqs, height_seqs, intensity_seqs], [ring_classification_seqs])

                        distance_cumsum  += np.sum(distance_seqs.flatten())
                        height_cumsum    += np.sum(height_seqs.flatten())
                        intensity_cumsum += np.sum(intensity_seqs.flatten())

                        if distance_mean is not None:
                            distance_cumdev2  += np.sum((distance_seqs.flatten() - distance_mean) ** 2)
                            height_cumdev2    += np.sum((height_seqs.flatten() - height_mean) ** 2)
                            intensity_cumdev2 += np.sum((intensity_seqs.flatten() - intensity_mean) ** 2)

                        yielded += batch_size
                        if yielded >= (len(items) * sector_splits):
                            _yielded = yielded * len(rings) * points_per_ring / sector_splits
                            if distance_mean is not None:
                                # we can now calculate variance (2nd pass)
                                distance_var  = distance_cumdev2 / _yielded
                                height_var    = height_cumdev2 / _yielded
                                intensity_var = intensity_cumdev2 / _yielded
                                if print_var:
                                    print('distance  var:  ' + str(distance_var) )
                                    print('height    var:  ' + str(height_var) )
                                    print('intensity var:  ' + str(intensity_var) )
                                    print_var = False
                                distance_cumdev2 = height_cumdev2 = intensity_cumdev2 = 0.

                            # calculate mean in the first pass
                            distance_mean  = distance_cumsum / _yielded
                            height_mean    = height_cumsum / _yielded
                            intensity_mean = intensity_cumsum / _yielded
                            if print_mean:
                                print('distance  mean: ' + str(distance_mean) )
                                print('height    mean: ' + str(height_mean) )
                                print('intensity mean: ' + str(intensity_mean) )
                                print_mean = False
                            yielded = 0
                            distance_cumsum = height_cumsum = intensity_cumsum = 0.

                    else:
                        yield ([lidars, distances], [centroids, box_sizes, yaws])

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

    # monkey-patch loss so model loads ok
    # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
    import keras.losses
    import keras.metrics
    keras.losses.angle_loss = angle_loss
    keras.metrics.min_angle_error_degrees = min_angle_error_degrees

    model = load_model(args.model)
    if args.pointnet:
        pointnet_points = model.get_input_shape_at(0)[0][1]
        points_per_ring = args.points_per_ring
        print('Loaded localizer model with ' + str(pointnet_points) + ' points')

    else:
        points_per_ring = model.get_input_shape_at(0)[0][1] * args.sector_splits
        nrings = model.get_input_shape_at(0)[0][2]
        print('Loaded segmenter model with ' + str(points_per_ring) + ' points per ring and ' + str(nrings) + ' rings')
        assert nrings == len(rings)

else:
    points_per_ring = args.points_per_ring
    if args.pointnet:
        pointnet_points = args.pointnet_points
        model = get_model_pointnet(pointnet_points)
    else:
        model = get_model_recurrent(
            points_per_ring,
            len(rings),
            hidden_neurons=args.hidden_neurons,
            sector_splits = args.sector_splits,
            bidirectional_first_pass = args.bidirectional_first_pass)

if (args.gpus > 1) or (len(args.batch_size) > 1):
    assert K._backend == 'tensorflow'
    import multi_gpu

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
            validate_items.extend(get_items(
                provider_didi.get_tracklets(R2_DATA_DIR, R2_VALIDATE_FILE, xml_filename=XML_TRACKLET_FILENAME_SAFE, box_scaling = BOX_SCALING),
                unsafe=False))
        if args.r3:
            validate_items.extend(get_items(
                provider_didi.get_tracklets(R3_DATA_DIR, R3_VALIDATE_FILE, xml_filename=XML_TRACKLET_FILENAME_UNSAFE, box_scaling = BOX_SCALING),
                unsafe=True))
    else:
        validate_items = get_items(
            provider_didi.get_tracklets(DATA_DIR, args.validate_file, xml_filename=XML_TRACKLET_FILENAME, box_scaling = BOX_SCALING),
            unsafe=UNSAFE_TRAINING)

if args.pointnet:
    _loss = ['mse', 'mse', angle_loss]
    _metrics = { 'centroid':'mae', 'box_size' : 'mae', 'yaw' : min_angle_error_degrees } #['mse', 'mse', 'mae']
else:
    _loss = 'binary_crossentropy'
    _metrics = { 'class':'acc'}


model.summary()
model.compile(loss=_loss, optimizer=Adam(lr=LEARNING_RATE) if args.pointnet else RMSprop(lr=LEARNING_RATE), metrics = _metrics)

if args.test:
    distance_seq = np.empty((points_per_ring, len(rings)), dtype=np.float32)
    if args.validate_split is not None:

        _items = get_items(
            provider_didi.get_tracklets(DATA_DIR, args.train_file, xml_filename=XML_TRACKLET_FILENAME, box_scaling = BOX_SCALING),
            unsafe=UNSAFE_TRAINING)
        _, validate_items = split_train(_items, test_size=args.validate_split * 0.01)

    _predictions = model.predict_generator(
        #        gen(train_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, localizer_points_per_ring),

        generator=gen(validate_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, args.sector_splits, training = False),
        steps= len(validate_items) // BATCH_SIZE)

    print(_predictions.shape)

    predictions =  _predictions.reshape((-1, points_per_ring, len(rings)))
    print(predictions.shape)
    i = 0
    for item,prediction in zip(validate_items,predictions):
        tracklet, frame = item
        lidar_d_i, lidar_int = tracklet.get_lidar_rings(frame,
                                             rings=rings,
                                             points_per_ring=points_per_ring,
                                             clip = CLIP_DIST,
                                             clip_h = CLIP_HEIGHT,
                                             return_lidar_interpolated=True)
        box = tracklet.get_box(frame).T

        for ring in rings:
            distance_seq[:, ring - rings[0]] = lidar_d_i[ring - rings[0], :, 0]

        point_idx_in_box = DidiTracklet.get_lidar_in_box(lidar_int, box.T, return_idx_only=True)
        ring_classification_seq = np.zeros((len(rings), points_per_ring), dtype=np.float32)
        ring_classification_seq[np.floor_divide(point_idx_in_box, points_per_ring), np.remainder(point_idx_in_box, points_per_ring)] = 1.

        save_lidar_plot(distance_seq, box, os.path.join('test', str(i) + '.png'),
                        highlight = None,
                        lidar_distance_gt=ring_classification_seq,
                        lidar_distance_pred=prediction)
        i += 1

else:
    if args.r2 or args.r3:
        train_items = [ ]
        if args.r2:
            train_items.extend(get_items(
                provider_didi.get_tracklets(R2_DATA_DIR, R2_TRAIN_FILE, xml_filename=XML_TRACKLET_FILENAME_SAFE, box_scaling = BOX_SCALING),
                unsafe=False))
        if args.r3:
            train_items.extend(get_items(
                provider_didi.get_tracklets(R3_DATA_DIR, R3_TRAIN_FILE, xml_filename=XML_TRACKLET_FILENAME_UNSAFE, box_scaling = BOX_SCALING),
                unsafe=True))
    else:
        train_items = get_items(
            provider_didi.get_tracklets(DATA_DIR, args.train_file, xml_filename=XML_TRACKLET_FILENAME, box_scaling = BOX_SCALING),
            unsafe=UNSAFE_TRAINING)
    if args.validate_split is not None:
        train_items, validate_items = split_train(train_items, test_size=args.validate_split * 0.01)

    print("Train items:    " + str(len(train_items)))
    print("Validate items: " + str(len(validate_items)))

    if args.pointnet:
        postfix = "-loc-rings_"+str(rings[0])+'_'+str(rings[-1]+1)
        metric  = "-val_loss{val_loss:.4f}"
        monitor = 'val_loss'
    else:
        postfix = "-seg-rings_"+str(rings[0])+'_'+str(rings[-1]+1)+'-sectors_'+str(args.sector_splits)
        metric  = "-val_class_acc{val_acc:.4f}"
        monitor = 'val_acc'

    modelsuffix = "" if args.model_suffix is None else "-" + args.model_suffix

    save_checkpoint = ModelCheckpoint(
        "lidarnet"+modelsuffix+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
        monitor=monitor,
        verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-7, epsilon = 0.0001, verbose=1)
    model.fit_generator(
        generator        = gen(train_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, args.sector_splits),
        steps_per_epoch  = len(train_items) // BATCH_SIZE,
        validation_data  = gen(validate_items, BATCH_SIZE, points_per_ring, rings, pointnet_points, args.sector_splits, training = False),
        validation_steps = len(validate_items) // BATCH_SIZE,
        epochs = args.max_epoch,
        callbacks = [save_checkpoint, reduce_lr])
