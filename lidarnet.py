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
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
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

R2_DATA_DIR      = '../release2/Data-points-processed'
R3_DATA_DIR      = '../release3/Data-points-processed'
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
parser.add_argument('-j', '--joint', action='store_true', help='Joint training of classification and localization')

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

def get_model_localizer(points_per_ring, rings, hidden_neurons):
    lidar_distances   = Input(shape=(points_per_ring, rings ))  # d i
    lidar_intensities = Input(shape=(points_per_ring, rings ))  # d i

    l0  = Lambda(lambda x: x * 1/50. , output_shape=(points_per_ring, rings))(lidar_distances)
    l1  = Lambda(lambda x: x * 1/128., output_shape=(points_per_ring, rings))(lidar_intensities)
    l0  = Reshape((points_per_ring, rings, 1))(l0)
    l1  = Reshape((points_per_ring, rings, 1))(l1)

    l  = Concatenate(axis=-1)([l0,l1])

    l = Conv2D(filters=hidden_neurons[0], kernel_size=(1, rings), activation='relu', padding='valid')(l)

    prev_n = hidden_neurons[0]
    for hidden_neuron in hidden_neurons:
        l  = Conv2D(filters=hidden_neuron,   kernel_size=(3,1), activation='relu', padding='same')(l)
        o  = l
        l  = Conv2D(filters=hidden_neuron/2, kernel_size=(1,1), activation='relu', padding='same')(l)
        l  = Conv2D(filters=hidden_neuron,   kernel_size=(3,1), activation='relu', padding='same')(l)
        l  = Add()([l,o])

    l = Conv2D(filters=rings   , kernel_size=(1, 1), activation='relu', padding='valid')(l)
    l = Conv2D(filters=rings//2, kernel_size=(1, 1), activation='relu', padding='valid')(l)

    l = MaxPooling2D(pool_size=(2,1), strides=(2,1))(l)
    l = Flatten()(l)
    l = Dense(64, activation='relu')(l)
    l = Dense(32, activation='relu')(l)
    l = Dense(16, activation='relu')(l)
    l = Dense(1)(l)
    distances = Lambda(lambda x: x * 50.)(l)
    outputs = [distances]

    model = Model(inputs=[lidar_distances, lidar_intensities], outputs=outputs)
    return model


def get_model_recurrent(points_per_ring, rings, hidden_neurons, localizer=False):
    lidar_distances   = Input(shape=(points_per_ring, rings ))  # d i
    lidar_intensities = Input(shape=(points_per_ring, rings ))  # d i

    l0  = Lambda(lambda x: x * 1/50. , output_shape=(points_per_ring, rings))(lidar_distances)
    l1  = Lambda(lambda x: x * 1/128., output_shape=(points_per_ring, rings))(lidar_intensities)

    o  = Concatenate(axis=-1, name='lidar')([l0,l1])

    l  = o

    if localizer:
        for i, hidden_neuron in enumerate(hidden_neurons):
            _return_sequences = True
            _activation = 'tanh'
            if (localizer is True) and (i == (len(hidden_neurons) - 1)):
                _return_sequences = False
                #_activation = 'linear'
            l = GRU(hidden_neuron, activation = _activation, return_sequences=_return_sequences, name='class-GRU'+str(i))(l)
        #l = Dense(1)(l)
        distances = l
        #distances = Lambda(lambda x: x * 50.)(l)
        outputs = [distances]
    else:
        for i, hidden_neuron in enumerate(hidden_neurons):
            _return_sequences = True
            _activation = 'tanh'
            _name = 'class-GRU' + str(i)
            if i == (len(hidden_neurons) - 1):
                _return_sequences = True
                _activation = 'sigmoid'
                _name = 'class'
            l = GRU(hidden_neuron, activation = _activation, return_sequences=_return_sequences, name=_name)(l)
        classification = l

        o = Concatenate(axis=-1)([o, classification])

        for i, hidden_neuron in enumerate(hidden_neurons):
            _return_sequences = True
            _activation = 'tanh'
            if i == (len(hidden_neurons) - 1):
                _return_sequences = False
                _activation = 'relu'
            o = GRU(hidden_neuron, activation = _activation, return_sequences=_return_sequences, name='loc-GRU'+str(i))(o)

        distance = Lambda(lambda x: x * 25., name='distance')(o)
        outputs = [classification, distance]

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

# if localizer_points_per_ring is a number => yield localizer semantics
def gen(items, batch_size, points_per_ring, rings, localizer_points_per_ring=None, training=True, ):

    angles         = np.empty((batch_size, 1), dtype=np.float32)
    distances      = np.empty((batch_size, 1), dtype=np.float32)
    centroids      = np.empty((batch_size, 3), dtype=np.float32)
    boxes          = np.empty((batch_size, 8, 3), dtype=np.float32)

    # inputs classifier
    distance_seqs  = np.empty((batch_size, points_per_ring, len(rings)), dtype=np.float32)
    intensity_seqs = np.empty((batch_size, points_per_ring, len(rings)), dtype=np.float32)

    # classifier label (output): 0 if point does not belong to car, 1 otherwise
    classification_seqs  = np.empty((batch_size, points_per_ring,1), dtype=np.float32)

    if localizer_points_per_ring is not None:
        l_distance_seqs  = np.empty((batch_size, localizer_points_per_ring, len(rings)), dtype=np.float32)
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
                    random_yaw = (np.random.random_sample() * 2. - 1.) * np.pi
                    centroid   = point_utils.rotZ(centroid, random_yaw)
                    box        = point_utils.rotZ(box, random_yaw)

                    flipX      = np.random.randint(2)
                    flipY      = np.random.randint(2)
                    if flipX:
                        centroid[0] = -centroid[0]
                        box[:,0]    = -box[:,0]
                    if flipY:
                        centroid[1] = -centroid[1]
                        box[:, 1] = -box[:, 1]

                    lidar_d_i  = tracklet.get_lidar_rings(frame,
                                                          rings = rings,
                                                          points_per_ring = points_per_ring,
                                                          clip=(0., 50.),
                                                          rotate = random_yaw,
                                                          flipX = flipX, flipY = flipY) #

                    # initialize with worst case to make sure they get updated
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

                    # car is between two frames, ignore it for now
                    if  angle_span >= np.pi:
                        skip = True
                    elif angle_span > max_angle_span:
                        max_angle_span = angle_span

                        #print("Max angle span:", max_angle_span)
                        #print(tracklet.xml_path, frame)
                        this_max_angle = True

            else:
                #print(tracklet, frame)
                lidar_d_i = tracklet.get_lidar_rings(frame,
                                                     rings = rings,
                                                     points_per_ring = points_per_ring,
                                                     clip = (0.,50.))  #

            if skip is False:

                for ring in rings:
                    distance_seqs [i, :, ring - rings[0]] = lidar_d_i[ring - rings[0], :, 0]
                    intensity_seqs[i, :, ring - rings[0]] = lidar_d_i[ring - rings[0], :, 1]

                angles[i]    = np.arctan2(centroid[1], centroid[0])
                centroids[i] = centroid
                distances[i] = distance
                boxes[i]     = box

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

                        if localizer_points_per_ring is None:
                            classification_seqs[ii, :] = 0.
                            classification_seqs[ii, _min_yaw:_max_yaw+1] = 1. # distances[ii] #for classification
                        else:
                            _yaw_span = _max_yaw - _min_yaw
                            for ring in rings:
                                l_distance_seqs [ii, :_yaw_span+1, ring - rings[0]] = distance_seqs [ii, _min_yaw:_max_yaw+1, ring - rings[0]]
                                l_intensity_seqs[ii, :_yaw_span+1, ring - rings[0]] = intensity_seqs[ii, _min_yaw:_max_yaw+1, ring - rings[0]]
                                l_distance_seqs [ii, _yaw_span+1:, ring - rings[0]] = 0.
                                l_intensity_seqs[ii, _yaw_span+1:, ring - rings[0]] = 0.

                            l_centroid_seqs[ii] = np.linalg.norm(centroids[ii, :2])

                    if localizer_points_per_ring is None:
                        yield ([distance_seqs, intensity_seqs], [classification_seqs, distances]) #)
                    else:
                        yield ([l_distance_seqs, l_intensity_seqs], l_centroid_seqs) #)

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
    points_per_ring = model.get_input_shape_at(0)[0][1]
    nrings = model.get_input_shape_at(0)[0][2]
    print('Loaded model with ' + str(points_per_ring) + ' points per ring and ' + str(nrings) + ' rings')
    assert points_per_ring == args.points_per_ring
    assert nrings == len(rings)
    localizer_points_per_ring = int(np.ceil(points_per_ring * args.worst_case_angle / (2 * np.pi ))) if args.localizer else None

else:
    points_per_ring = args.points_per_ring
    localizer_points_per_ring = int(np.ceil(points_per_ring * args.worst_case_angle / (2 * np.pi ))) if args.localizer else None

    if args.localizer:
        model = get_model_recurrent(localizer_points_per_ring, len(rings), hidden_neurons=args.hidden_neurons, localizer=True)
    else:
        model = get_model_recurrent(localizer_points_per_ring if args.localizer else points_per_ring, len(rings), hidden_neurons=args.hidden_neurons, localizer=args.localizer)

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

if args.localizer:
    _loss = 'mse'
    _metrics = ['mse']
else:
    _loss = ['binary_crossentropy', 'mse'] if args.joint else ['binary_crossentropy', null_loss]
    _metrics = ['acc', 'mse']

# set weights for the localizer part of the network according to args.joint
for i, hn in enumerate(args.hidden_neurons):
    model.get_layer(name='loc-GRU'+str(i)).trainable = args.joint

model.summary()
model.compile(loss=_loss, optimizer=RMSprop(lr=LEARNING_RATE, ), metrics = _metrics)

if args.test:
    distance_seq = np.empty((points_per_ring, len(rings)), dtype=np.float32)
    if args.validate_split is not None:
        _items = get_items(provider_didi.get_tracklets(DATA_DIR, args.train_file, xml_filename=XML_TRACKLET_FILENAME), unsafe=UNSAFE_TRAINING)
        _, validate_items = split_train(_items, test_size=args.validate_split * 0.01)

    predictions = model.predict_generator(
        generator=gen(validate_items, BATCH_SIZE, points_per_ring, rings, training = False),
        steps=len(validate_items) // BATCH_SIZE)

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

    if args.localizer:
        postfix = "-loc-rings_"+str(rings[0])+'_'+str(rings[-1]+1)
        metric  = "-val_loss{val_loss:.4f}"
        monitor = 'val_loss'
    else:
        postfix = "-cla-rings_"+str(rings[0])+'_'+str(rings[-1]+1)
        metric  = "-val_class_acc{val_class_acc:.4f}"
        monitor = 'val_class_acc'

    save_checkpoint = ModelCheckpoint(
        "lidarnet"+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
        monitor=monitor,
        verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-7, epsilon = 0.0001, verbose=1)

    model.fit_generator(
        gen(train_items, BATCH_SIZE, points_per_ring, rings, localizer_points_per_ring),
        steps_per_epoch  = len(train_items) // BATCH_SIZE,
        validation_data  = gen(validate_items, BATCH_SIZE, points_per_ring, rings, localizer_points_per_ring, training = False),
        validation_steps = len(validate_items) // BATCH_SIZE,
        epochs = args.max_epoch,
        callbacks = [save_checkpoint, reduce_lr])
