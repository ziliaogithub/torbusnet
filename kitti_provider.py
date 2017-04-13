import numpy as np
import kitti_utils
from scipy.misc import imsave
import re
import os
import h5py

#basedir = '../../kitti/'
#date = '2011_09_26'
#drive = '0013'

def parse_filename(filename):
    match = re.search(r'(.*)/\d{4}_\d\d_\d\d/(\d{4}_\d\d_\d\d)_drive_(\d{4})_sync/tracklet_labels.xml', os.path.abspath(filename))
    basedir = match.group(1)
    date    = match.group(2)
    drive   = match.group(3)
    return (basedir, date, drive)

def generate_top_views(
    cur_data, 
    cur_seg, 
    pred_seg_res, 
    cur_test_filename,
    MODEL_STORAGE_PATH,
    begidx):

    print("GT   Object points:", np.sum(cur_seg))
    print("Pred Object points:", np.sum(pred_seg_res))

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


def load_h5_data_label_seg(h5_filename):

    MAX_POINTS = 16384

    h5 = h5_filename.replace(".xml", ".h5")
    if os.path.isfile(h5):
        h5f = h5py.File(h5,'r')
        data  = h5f['data'][:]
        label = h5f['label'][:]
        seg   = h5f['seg'][:]
        print(data.shape, label.shape, seg.shape)
        assert data.shape[1:3] == (MAX_POINTS,3)
        assert label.shape[1]  == 1
        assert seg.shape[1]    == (MAX_POINTS,3)
        assert data.shape[0]   == label.shape[0] == seg.shape[0]
        return (data, label, seg)

    (basedir, date, drive) = parse_filename(h5_filename)
    print("Loading", basedir, date, drive)
    parser = kitti_utils.Parser(basedir, date, drive)
    frames = parser.frames(only_with = ['Car', 'Van', 'Truck', 'Pedestrian', 'Sitter', 'Cyclist', 'Tram', 'Misc', 'Person (sitting)']) 
    lidars = []
    tracked_objects = []
    
    data  = np.empty((len(frames), MAX_POINTS, 3), dtype=np.float32)
    label = np.zeros((len(frames), 1), dtype=np.uint8)
    seg   = np.empty((len(frames), MAX_POINTS), dtype=np.uint8)

    fi = 0
    for frame in frames:
        lidar_with_label = parser.lidar_with_label(frame, MAX_POINTS=MAX_POINTS) # returns (M, 5)
        lidar_points = lidar_with_label.shape[0]
        if lidar_points > MAX_POINTS:
            #import code
            #code.interact(local=locals())
            lidar_with_label = lidar_with_label[lidar_with_label[:,0].argsort()][:MAX_POINTS,:] # np.sort(lidar_with_label, axis=0)[:MAX_POINTS,:]
            #top_view = parser.top_view(frame, with_boxes = True, lidar_override=lidar_with_label[:,0:3])
            #imsave('tv{}.png'.format(frame), top_view)
        elif lidar_points < MAX_POINTS:
            missing = MAX_POINTS - lidar_points
            print("Super sampling", missing,"points from", lidar_with_label.shape[0], "to", MAX_POINTS)
            lidar_with_label = np.concatenate((lidar_with_label,lidar_with_label[np.random.choice(range(lidar_points), missing, replace=False)]), axis = 0)

        data[fi,:] = lidar_with_label[:,0:3]
        seg[fi,:]  = lidar_with_label[:,4]
        fi += 1

    h5f = h5py.File(h5, 'w')
    h5f.create_dataset('data',  data=data)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('seg',   data=seg)
    h5f.close()

    # data (B, points, 3)
    # label B, 1) uint8
    # seg (B, points, 3) uint8
    return (data, label, seg)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)