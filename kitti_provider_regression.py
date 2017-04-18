import numpy as np
import kitti_utils
from scipy.misc import imsave
import re
import os
import h5py

#basedir = '../../kitti/'
#date = '2011_09_26'
#drive = '0013'

kitti_categories = kitti_utils.Parser.kitti_cat_names

def parse_filename(filename):
    match = re.search(r'(.*)/\d{4}_\d\d_\d\d/(\d{4}_\d\d_\d\d)_drive_(\d{4})_sync/tracklet_labels.xml', os.path.abspath(filename))
    basedir = match.group(1)
    date    = match.group(2)
    drive   = match.group(3)
    return (basedir, date, drive)

total_a = 0
total_b = 0

def generate_top_views_with_boxes(
    cur_data,
    cur_boxes,
    pred_boxes,
    cur_test_filename,
    MODEL_STORAGE_PATH):
    print("GT A,B,l", cur_boxes[0])
    print("Pr A,B,l", pred_boxes[0])
    return

def generate_top_views(
    cur_data, 
    cur_seg, 
    pred_seg_res, 
    cur_test_filename,
    MODEL_STORAGE_PATH,
    begidx,
    num_parts):
    global total_a, total_b
    print("GT mean:", np.mean(cur_data[0,:,0]),np.mean(cur_data[0,:,1]),np.mean(cur_data[0,:,2]))
    print("GT   Object points:", np.sum(cur_seg), "eg:", cur_seg[0][0:2048:256])
    print(np.bincount(cur_seg[0], minlength=num_parts))
    print("Pred Object points:", np.sum(pred_seg_res), pred_seg_res[0][0:2048:256])
    print(np.bincount(pred_seg_res[0], minlength=num_parts))
    hist_gt = np.bincount(cur_seg.flatten())
    total_a += hist_gt[0]
    total_b += hist_gt[1]
    print(total_a, "/", total_b, total_a+total_b)

    X_RANGE = (  0., 70.)
    Y_RANGE = (-40., 40.)
    Z_RANGE = ( -2.,  2.)
    RES = 0.2

    Y_PIXELS = int((Y_RANGE[1]-Y_RANGE[0]) / RES)
    X_PIXELS = int((X_RANGE[1]-X_RANGE[0]) / RES)

    top_view = np.zeros(shape=(X_PIXELS, Y_PIXELS, 3),dtype=np.float32)
    
    # convert from lidar x y to top view X Y 
    def toY(y):
        return int((y-Y_RANGE[0]) // RES)
    def toX(x):
        return int((X_PIXELS-1) - (x-X_RANGE[0]) // RES)
    def toXY(x,y):
        return (toY(y), toX(x))

    (basedir, date, drive) = parse_filename(cur_test_filename)

    for (point_batch, label_batch, prediction_batch) in zip(cur_data, cur_seg, pred_seg_res):
        for (point, label, prediction) in zip(point_batch, label_batch, prediction_batch):
            x, y = point[0], point[1]
            if (x >= X_RANGE[0]) and (x <= X_RANGE[1]) and (y >= Y_RANGE[0]) and (y <= Y_RANGE[1]):
                if (label == prediction):
                    if label == 1:
                        C = (1.,  1.,   1.)
                    else:
                        C = (0.5, 0.5, 0.5)
                else:
                    if (label == 0) and (prediction == 1):
                        C = (1., 0.,  0.)
                    else:
                        C = (1., 0.,  1.)

                top_view[toXY(x,y)[::-1] ] = C

        imsave('{}/drive{}-{}.png'.format(MODEL_STORAGE_PATH, drive, begidx), top_view)
        begidx += 1
        top_view[:,:] = (0,0,0)
    return

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def normalize(data):
    return data
    for (i,batch) in enumerate(data):
        data[i,:,0] -= np.mean(batch[:,0])
        data[i,:,1] -= np.mean(batch[:,1])
        data[i,:,2] -= np.mean(batch[:,2])
    return data

def load_h5_data_label_seg(h5_filename, MAX_POINTS, MIN_POINTS=20, IMAGE_STORAGE_PATH="./images"):

    h5 = h5_filename.replace(".xml", ".h5")
    if True and os.path.isfile(h5):
        h5f = h5py.File(h5,'r')
        data  = h5f['data'][:]
        label = h5f['label'][:]
        boxes = h5f['boxes'][:]
        # shuffle point clouds
        for point_cloud in data:
            np.random.shuffle(point_cloud)    
        assert data.shape[1:3] == (MAX_POINTS,3)
        #assert label.shape[1]  == 1

        return (normalize(data), label, boxes)

    (basedir, date, drive) = parse_filename(h5_filename)
    print("Loading", basedir, date, drive)
    parser = kitti_utils.Parser(basedir, date, drive)
    frames = parser.frames(only_with = ['Car']) 
    lidars = []
    tracked_objects = []
    
    data  = np.empty((0, MAX_POINTS, 3), dtype=np.float32)
    label = np.empty((0), dtype=np.uint8)
    boxes = np.empty((0, 5), dtype=np.float32)

    for frame in frames:
        if True:
            lidar_cones, image_with_lidar_cones = parser.lidar_cone_of_detected_objects(frame, return_image=True)
            if not os.path.exists(IMAGE_STORAGE_PATH):
                os.mkdir(IMAGE_STORAGE_PATH)
            imsave('{}/cone-drive{}-{}.png'.format(IMAGE_STORAGE_PATH, drive, frame), image_with_lidar_cones)
        else:
            lidar_cones = parser.lidar_cone_of_detected_objects(frame, return_image=False)

        for (lidar_in_2d_bbox, bbox, cat_idx) in lidar_cones:
            lidar_points = lidar_in_2d_bbox.shape[0]
            # discard intensity 
            lidar_in_2d_bbox = lidar_in_2d_bbox[:,0:3]
            print(lidar_points, "found in", bbox)
            if lidar_points >= MIN_POINTS:
                if lidar_points > MAX_POINTS:
                    lidar_in_2d_bbox = parser._lidar_subsample(lidar_in_2d_bbox, MAX_POINTS)
                elif lidar_points < MAX_POINTS:
                    missing = MAX_POINTS - lidar_points
                    lidar_in_2d_bbox = np.concatenate((lidar_in_2d_bbox,lidar_in_2d_bbox[np.random.choice(range(lidar_points), missing, replace=True)]), axis = 0)
                np.random.shuffle(lidar_in_2d_bbox)
                data  = np.concatenate((data,  np.expand_dims(lidar_in_2d_bbox, axis=0)), axis=0)
                label = np.append(label, [cat_idx], axis=0)
                boxes = np.concatenate((boxes, [bbox[0:5]]), axis=0)

    label = np.expand_dims(label, axis=0)

    if True:
        h5f = h5py.File(h5, 'w')
        h5f.create_dataset('data',  data=data)
        h5f.create_dataset('label', data=label)
        h5f.create_dataset('boxes', data=boxes)
        h5f.close()

    # data (B, points, 3)
    # label B, 1) uint8
    # seg (B, points) uint8
    # e.g. (1870, 2048, 3) (1870, 1) (1870, 2048)
    return (normalize(data), label, boxes)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(filename, MAX_POINTS=1024):
    return load_h5_data_label_seg(filename, MAX_POINTS)