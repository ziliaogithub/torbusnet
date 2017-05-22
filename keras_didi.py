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
parser.add_argument('--num_point', type=int, default=24000, help='Number of lidar points to use')  #real number per lidar cycle is 32000, we will reduce to 16000
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run')
parser.add_argument('--max_dist', type=float, default=25, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('--max_dist_offset', type=float, default=3, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--optimizer', default='adam', help='adam or momentum ')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs used for training')
parser.add_argument('-c', '--classifier', action='store_true', help='Train classifier instead of regressor')

args = parser.parse_args()

BATCH_SIZE     = args.batch_size
NUM_POINT      = args.num_point
MAX_EPOCH      = args.max_epoch
LEARNING_RATE  = args.learning_rate
OPTIMIZER      = args.optimizer
DATA_DIR       = args.data_dir
MAX_DIST       = args.max_dist
CLASSIFIER     = args.classifier
MAX_LIDAR_DIST = MAX_DIST + args.max_dist_offset

def get_model_functional(classifier=False):
    points = Input(shape=(NUM_POINT, 4))

    p = Lambda(lambda x: x * (1. / 25., 1. / 25., 1. / 3., 1. / 64.) - (0., 0., -0.5, 1.))(points)
    p = Reshape(target_shape=(NUM_POINT, 4, 1), input_shape=(NUM_POINT, 4))(p)
    p = Conv2D(filters=  64, kernel_size=(1, 4), activation='relu')(p)
    po = p
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu')(p)

    #p  = TorbusMaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')([p, po])
    p  = MaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')(p)

    p  = Flatten()(p)
    p  = Dense(512, activation='relu')(p)

    if classifier is False:

        pc = Dense(256, activation='relu')(p)
        pc = Dropout(0.3)(pc)
        c = Dense(3, activation=None)(pc)

        ps = Dense( 32, activation='relu')(p)
        s = Dense(3, activation=None)(ps)

        centroids  = Lambda(lambda x: x * (25.,25., 3.) - (0., 0., -1.5))(c) # tx ty tz
        dimensions = Lambda(lambda x: x * ( 3.,25.,25.) - (-1.5, 0., 0.))(s) # h w l
        model = Model(inputs=points, outputs=[centroids, dimensions])

    else:

        p = Dropout(0.3)(p)
        p = Dense(256, activation='relu')(p)
        p = Dropout(0.3)(p)
        c = Dense(1, activation='sigmoid')(p)

        model = Model(inputs=points, outputs=c)
    return model

if args.model:
    print("Loading model " + args.model)
    model = load_model(args.model)
    model.summary()
else:
    model = get_model_functional(classifier = CLASSIFIER)
    model.summary()

if args.gpus > 1:
    model = multi_gpu.make_parallel(model, args.gpus )
    BATCH_SIZE *= args.gpus

if CLASSIFIER is False:
    model.compile(loss='mse', optimizer=Nadam(lr=LEARNING_RATE))
else:
    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=LEARNING_RATE), metrics=['accuracy'])


# -----------------------------------------------------------------------------------------------------------------
def gen(items, batch_size, num_points, training=True, classifier=False):
    lidars      = np.empty((batch_size, num_points, 4))
    centroids   = np.empty((batch_size, 3))
    dimensions  = np.empty((batch_size, 3))

    classifications = np.empty((batch_size,1), dtype=np.int32)

    BINS = 25.

    if training is False:
        avgs = mins = maxs = 0.
        for item in items:
            tracklet, frame = item
            lidar = tracklet.get_lidar(frame, num_points, max_distance=MAX_LIDAR_DIST)[:, :4]
            avgs += np.mean(lidar, axis=0)
            mins += np.amin(lidar, axis=0)
            maxs += np.amax(lidar, axis=0)
        avgs /= len(items)
        mins /= len(items)
        maxs /= len(items)

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
                    random_yaw   = (np.random.random_sample() * 2. - 1.) * np.pi
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
                    random_t = np.zeros(4) # (np.random.random_sample((4)) - np.ones(4)/2. ) * np.array([MAX_DIST/BINS, MAX_DIST/BINS, 0., 0.])
                    perturbation  = (np.random.random_sample((lidar.shape[0], 4)) - np.ones(4)/2. ) * np.array([0.1, 0.1, 0.1, 4.])

                    lidar        += random_t + perturbation
                    centroid[:3] += random_t[:3]

            else:
                lidar = tracklet.get_lidar(frame, num_points, max_distance=MAX_LIDAR_DIST)[:, :4]

            if skip is False:
                if classifier:
                    OBS_DIAG   = 2.5
                    ALLOWED_ERROR = 4.
                    CLASS_DIST = OBS_DIAG + ALLOWED_ERROR
                    classification = np.random.randint(2)
                    num_points = lidar.shape[0]
                    class_points = 0
                    attempts = 0
                    while class_points < 1:
                        if classification == 0:
                            # generate a non-detection by selecting an area where is no car:
                            # find a random point within MAX_DIST of center that is at least CLASS_DIST + OBS_DIAG from vehicle
                            random_center = np.array(centroid[:2])
                            while ((np.linalg.norm(random_center - centroid[:2]) <= (CLASS_DIST + OBS_DIAG)) or
                                (np.linalg.norm(random_center) >= MAX_DIST)):
                                random_center = (np.random.random_sample(2) * 2. - np.ones(2)) * MAX_DIST
                            classification_center = random_center
                        else:
                            # for detections we want to jitter the provided centroid:
                            # put classification centroid within ALLOWED_ERROR of real centroid
                            dist = ALLOWED_ERROR + 1.
                            while dist > ALLOWED_ERROR:
                                classification_center = centroid[:2] + (2 * np.random.random_sample(2) - np.ones(
                                    2)) * ALLOWED_ERROR
                                dist = np.linalg.norm(centroid[:2] - classification_center)
                        class_lidar = lidar.copy()
                        class_lidar[:, :2] -= classification_center
                        class_lidar  = class_lidar[( (class_lidar[:,0] ** 2) + (class_lidar[:,1] ** 2) ) <= (CLASS_DIST ** 2)]
                        class_points = class_lidar.shape[0]
                        attempts += 1
                        #if attempts > 2:
                        #    print(classification, classification_center, class_points, tracklet.xml_path, frame)

                    # random rotate so we see all yaw equally
                    random_yaw   = (np.random.random_sample() * 2. - 1.) * np.pi
                    class_lidar        = point_utils.rotZ(class_lidar, random_yaw)
                    lidar_size = class_lidar.shape[0]
                    class_lidar = np.concatenate((class_lidar, class_lidar[np.random.choice(lidar_size, size=num_points - lidar_size, replace=True)]), axis=0)

                    lidars[i] = class_lidar
                    classifications[i] = classification
                    i += 1
                    if i == batch_size:
                        yield (lidars, classifications)
                        i = 0
                        #print(classifications)

                else: # regression
                    lidars[i]     = lidar
                    centroids[i]  = centroid
                    dimensions[i] = dimension

                    i += 1
                    if i == batch_size:
                        yield (lidars, [centroids, dimensions])
                        i = 0
                        if training:
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

if CLASSIFIER:
    postfix = "classifier"
    metric  = "-acc{val_acc:.4f}"
else:
    postfix = "regressor"
    metric  = "-loss{val_loss:.2f}"

save_checkpoint = ModelCheckpoint(
    "torbusnet-"+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
    monitor='val_acc' if CLASSIFIER else 'val_loss',
    verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=5e-7, epsilon = 0.2, cooldown = 10)

model.fit_generator(
    gen(train_items, BATCH_SIZE, NUM_POINT, classifier=CLASSIFIER),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validate_items, BATCH_SIZE, NUM_POINT, training=False, classifier=CLASSIFIER),
    validation_steps = len(validate_items) // BATCH_SIZE,
    epochs = 2000,
    callbacks = [save_checkpoint, reduce_lr])