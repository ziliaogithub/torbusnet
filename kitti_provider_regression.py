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

def generate_top_views_with_boxes(
    cur_data,
    cur_boxes,
    pred_boxes,
    cur_test_filename,
    MODEL_STORAGE_PATH,
    IMAGE_STORAGE_PATH="./images",
    idx=0):
    print("GT A,B,l", cur_boxes[0])
    print("Pr A,B,l", pred_boxes[0])
    (basedir, date, drive) = parse_filename(cur_test_filename)
    print("Dummy loading", basedir, date, drive)
    parser = kitti_utils.Parser(basedir, date, drive)
    if not os.path.exists(IMAGE_STORAGE_PATH):
        os.mkdir(IMAGE_STORAGE_PATH)

    img = parser.top_view(0, with_boxes = False, lidar_override = np.squeeze(cur_data, axis=0), abl_overrides=np.concatenate((cur_boxes, pred_boxes), axis=0))
    imsave('{}/GTvsPred{}-{}.png'.format(IMAGE_STORAGE_PATH, drive, idx), img)

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

        if False:
            (basedir, date, drive) = parse_filename(h5_filename)
            print("Dummy loading", basedir, date, drive)
            parser = kitti_utils.Parser(basedir, date, drive)
            ii = 0
            if not os.path.exists(IMAGE_STORAGE_PATH):
                os.mkdir(IMAGE_STORAGE_PATH)
            for abl, lidar in zip(boxes, data):
                img = parser.top_view(0, with_boxes = False, lidar_override = lidar, abl_overrides=[abl])
                imsave('{}/GT{}-{}.png'.format(IMAGE_STORAGE_PATH, drive, ii), img)
                ii += 1

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