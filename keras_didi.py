import provider_didi
import argparse
from keras.initializers import Constant, Zeros
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model, Input
from keras.layers import Input, merge, Layer, concatenate
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
parser.add_argument('--num_point', type=int, default=2048, help='Number of lidar points to use')  #real number per lidar cycle is 32000, we will reduce to 16000
parser.add_argument('--classification_points', type=int, default=2048, help='Number of lidar points to use')  #real number per lidar cycle is 32000, we will reduce to 16000
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run')
parser.add_argument('--max_dist', type=float, default=25, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('--max_dist_offset', type=float, default=3, help='Ignore centroids beyond this distance (meters)')
parser.add_argument('-b', '--batch_size', type=int, nargs='+', default=[12], help='Batch Size during training, or list of batch sizes for each GPU, e.g. -b 12,8')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-2, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs used for training')
parser.add_argument('-c', '--classifier', action='store_true', help='Train classifier instead of regressor')

args = parser.parse_args()

#NUM_POINT      = args.num_point
MAX_EPOCH      = args.max_epoch
LEARNING_RATE  = args.learning_rate
DATA_DIR       = args.data_dir
MAX_DIST       = args.max_dist
CLASSIFIER     = args.classifier
MAX_LIDAR_DIST = MAX_DIST + args.max_dist_offset

REGRESSION_POINTS = args.classification_points
CLASSIFICATION_POINTS = args.classification_points

assert args.gpus  == len(args.batch_size)

def get_model_regression():
    points = Input(shape=(REGRESSION_POINTS, 4))

    p = Lambda(lambda x: x * (1. / 10., 1. / 10., 1. / 3., 1. / 64.) - (0., 0., -0.5, 1.))(points)
    p = Reshape(target_shape=(REGRESSION_POINTS, 4, 1), input_shape=(REGRESSION_POINTS, 4))(p)
    p = Conv2D(filters=  64, kernel_size=(1, 4), activation='relu')(p)
    po = p
    p = Conv2D(filters= 128, kernel_size=(1, 1), dilation_rate = (1,1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), dilation_rate = (1,1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), dilation_rate = (1,1), activation='relu')(p)
    p = Conv2D(filters=1024, kernel_size=(1, 1), dilation_rate = (1,1), activation='relu')(p)

    #p  = TorbusMaxPooling2D(pool_size=(NUM_POINT, 1), strides=None, padding='valid')([p, po])
    p  = MaxPooling2D(pool_size=(REGRESSION_POINTS, 1), strides=None, padding='valid')(p)

    p  = Flatten()(p)
    p  = Dense(512, activation='relu')(p)
    pc = Dense(256, activation='relu')(p)
    pc = Dense( 32, activation='relu')(p)
    pc = Dropout(0.3)(pc)
    c = Dense(3, activation=None)(pc)
    ps = Dense( 32, activation='relu')(p)
    s = Dense(3, activation=None)(ps)

    centroids  = Lambda(lambda x: x * (25.,25., 3.) - (0., 0., -1.5))(c) # tx ty tz
    dimensions = Lambda(lambda x: x * ( 3.,25.,25.) - (-1.5, 0., 0.))(s) # h w l

    model = Model(inputs=points, outputs=[centroids, dimensions])
    return model

def detection_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)

    conf_loss = K.mean(K.binary_crossentropy(y_pred[:,0], y_true[:,0]), axis=-1)

    return conf_loss

def detection_accuracy(y_true, y_pred):

    conf_accuracy =  K.mean(K.equal(y_true[:,0], K.round(y_pred[:,0])), axis=-1)

    return conf_accuracy

def get_model_classification():
    points = Input(shape=(CLASSIFICATION_POINTS, 4))

    p = Lambda(lambda x: x * (1. / 25., 1. / 25., 1. / 3., 1. / 64.) - (0., 0., -0.5, 1.))(points)
    p = Lambda(lambda x: x - K.mean(x, axis=1, keepdims = True))(p)
    p = Reshape(target_shape=(CLASSIFICATION_POINTS, 4, 1), input_shape=(CLASSIFICATION_POINTS, 4))(p)
    p = Conv2D(filters=  64, kernel_size=(1, 4), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 128, kernel_size=(1, 1), activation='relu')(p)
    p = Conv2D(filters= 256, kernel_size=(1, 1), activation='relu')(p)

    p = MaxPooling2D(pool_size=(CLASSIFICATION_POINTS, 1), strides=None, padding='valid')(p)


    p  = Flatten()(p)
    p  = Dense(128, activation='relu')(p)

    #p = Dropout(0.3)(p)
    p = Dense(64, activation='relu')(p)
    p = Dense(32, activation='relu')(p)

    #p = Dropout(0.3)(p)
    classification = Dense(1, activation='sigmoid', name='classification')(p)
    #centroid = Dense(3)(p)
#    mean = K.squeeze(mean, axis=1)
#    centroid = Lambda(lambda x: x + mean[:,:3])(centroid)
    #centroid = Lambda(lambda x: x * (25., 25., 3.) + (0., 0., -1.5),  name='centroid')(centroid)

    #detection = concatenate([classification, centroid], name='detection')

    model = Model(inputs=points, outputs=classification)

    #model = Model(inputs=points, outputs=[classification, centroid])
    return model

if args.model:
    print("Loading model " + args.model)
    model = load_model(args.model)
    model.summary()
else:
    model = get_model_classification() if args.classifier else get_model_regression()
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

if CLASSIFIER is False:
    model.compile(loss='mse', optimizer=Nadam(lr=LEARNING_RATE))
else:
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=LEARNING_RATE), metrics=['accuracy'])


# -----------------------------------------------------------------------------------------------------------------
def gen(items, batch_size, num_points, training=True, classifier=False):
    lidars      = np.empty((batch_size, num_points, 4))
    centroids   = np.empty((batch_size, 3))
    dimensions  = np.empty((batch_size, 3))

    classifications      = np.empty((batch_size, 1), dtype=np.int32)
    relative_centroids   = np.empty((batch_size, 3))

    detections           = np.empty((batch_size, 4))

    BINS = 25.

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
                        h_current[:] = h_target[:]

                    lidar = tracklet.get_lidar(frame, max_distance=MAX_LIDAR_DIST)[:, :4]

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
                    perturbation  = (np.random.random_sample((lidar.shape[0], 4)) - np.ones(4)/2. ) * np.array([0.2, 0.2, 0.1, 4.])

                    lidar        += random_t + perturbation
                    centroid[:3] += random_t[:3]

            else:
                lidar = tracklet.get_lidar(frame, max_distance=MAX_LIDAR_DIST)[:, :4]

            if skip is False:

<<<<<<< Updated upstream
                OBS_DIAG   = 2.5
                ALLOWED_ERROR = 10.
                CLASS_DIST = OBS_DIAG + ALLOWED_ERROR
                classification = np.random.randint(2)
                class_points = 0
                attempts = 0
                while class_points < 1:
                    if (classification == 1) or (classifier is False):
                        # for detections we want to jitter the provided centroid:
                        # put classification centroid within ALLOWED_ERROR of real centroid
                        dist = ALLOWED_ERROR + 1.
                        while dist > ALLOWED_ERROR:
                            classification_center = centroid[:2] + (2 * np.random.random_sample(2) - np.ones(2)) * ALLOWED_ERROR
                            dist = np.linalg.norm(centroid[:2] - classification_center)
                    else:
                        # generate a non-detection centroid by selecting an area where is no car:
                        # find a random point within MAX_DIST of center that is at least CLASS_DIST + OBS_DIAG from vehicle
                        random_center = np.array(centroid[:2])
                        while ((np.linalg.norm(random_center - centroid[:2]) <= (CLASS_DIST + OBS_DIAG)) or
                                   (np.linalg.norm(random_center) >= MAX_DIST)):
                            random_center = (np.random.random_sample(2) * 2. - np.ones(2)) * MAX_DIST
                        classification_center = random_center

                    class_lidar = lidar
                    class_lidar[:, :2] -= classification_center
                    class_lidar  = class_lidar[( (class_lidar[:,0] ** 2) + (class_lidar[:,1] ** 2) ) <= (CLASS_DIST ** 2)]
                    class_points = class_lidar.shape[0]
                    attempts += 1
                    if attempts >= 2:
                        print(classification, classification_center, class_points, tracklet.xml_path, frame)

                # random rotate so we see all yaw equally
                random_yaw  = (np.random.random_sample() * 2. - 1.) * np.pi
                class_lidar           = point_utils.rotZ(class_lidar, random_yaw)
                classification_center = point_utils.rotZ(classification_center, random_yaw)
                class_lidar = DidiTracklet.resample_lidar(class_lidar, num_points)
                #print(num_points)
                lidars[i] = class_lidar
                classifications[i] = classification
                centroid[:2] = classification_center
                centroids[i]  = centroid
                dimensions[i] = dimension

                i += 1
                if i == batch_size:
                    if classifier:
                        yield (lidars, classifications)
                    else:
=======
                if classifier:
                    OBS_DIAG   = 2.5
                    ALLOWED_ERROR = 10.
                    CLASS_DIST = OBS_DIAG + ALLOWED_ERROR
                    classification = np.random.randint(2)
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
                                classification_center = \
                                    centroid[:2] + \
                                    (2 * np.random.random_sample(2) - np.ones(2)) * ALLOWED_ERROR
                                dist = np.linalg.norm(centroid[:2] - classification_center)
                        class_lidar = lidar.copy()
                        class_lidar[:, :2] -= classification_center
                        class_lidar  = class_lidar[( (class_lidar[:,0] ** 2) + (class_lidar[:,1] ** 2) ) <= (CLASS_DIST ** 2)]
                        class_points = class_lidar.shape[0]
                        attempts += 1
                        if attempts > 2:
                            print(classification, classification_center, class_points, tracklet.xml_path, frame)

                    relative_centroid = centroid
                    relative_centroid[:2] -= classification_center

                    # random rotate so we see all yaw equally
                    random_yaw  = (np.random.random_sample() * 2. - 1.) * np.pi
                    class_lidar = point_utils.rotZ(class_lidar, random_yaw)
                    class_lidar = DidiTracklet.resample_lidar(class_lidar, num_points)

                    relative_centroid     = point_utils.rotZ(relative_centroid, random_yaw)
                    lidars[i]             = class_lidar
                    classifications[i]    = classification
                    relative_centroids[i] = relative_centroid

                    detections[i]         = np.concatenate((np.array([classification], dtype=np.float32), relative_centroid), axis=0)

                    i += 1
                    if i == batch_size:
                        yield (lidars, classifications)
                        #yield(lidars, detections)
                        i = 0
                        #print(classifications)

                else: # regression
                    lidars[i]     = lidar
                    centroids[i]  = centroid
                    dimensions[i] = dimension

                    i += 1
                    if i == batch_size:
>>>>>>> Stashed changes
                        yield (lidars, [centroids, dimensions])

                    i = 0
                    #print(classifications)


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
    metric  = "-acc{val_loss:.4f}"
else:
    postfix = "regressor"
    metric  = "-val-loss{val_loss:.2f}"

save_checkpoint = ModelCheckpoint(
    "torbusnet-"+postfix+"-epoch{epoch:02d}"+metric+".hdf5",
    monitor='loss' if CLASSIFIER else 'val_loss',
    verbose=0,  save_best_only=True, save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7, epsilon = 0.2, cooldown = 5, verbose=1)

nn_points = CLASSIFICATION_POINTS# if args.classifier else REGRESSION_POINTS

model.fit_generator(
    gen(train_items, BATCH_SIZE, num_points=nn_points, classifier=CLASSIFIER),
    steps_per_epoch  = len(train_items) // BATCH_SIZE,
    validation_data  = gen(validate_items, BATCH_SIZE, num_points=nn_points, classifier=CLASSIFIER, training = False),
    validation_steps = len(validate_items) // BATCH_SIZE,
    epochs = 2000,
    callbacks = [reduce_lr])