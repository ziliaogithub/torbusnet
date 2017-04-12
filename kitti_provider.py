import pykitti
import numpy as np
import kitti_utils
from scipy.misc import imsave

basedir = '../../kitti/'
date = '2011_09_26'
drive = '0013'

start_frame  = 20
total_frames = 1


def load_h5_data_label_seg(h5_filename):
    #f = h5py.File(h5_filename)
    #data = f['data'][:]
    #label = f['label'][:]
    #seg = f['pid'][:]



    # dummy load to get calibration dat


    parser = kitti_utils.Parser(basedir, date, drive)
    frames = parser.frames(only_with = ['Car', 'Truck', 'Pedestrian', 'Sitter', 'Cyclist', 'Tram', 'Misc']) 
    lidars = []
    tracked_objects = []
    for frame in frames:
        #lidar = parser.lidar(frame)
        #lidars.append(lidar)
        #tracked_objects = parser.tracked_objects(frame)
        top_view = parser.top_view(frame, with_boxes = True)
        imsave('tv{}.png'.format(frame), top_view)

    # list of frames with objects detected # array of (points,3)
    # data (B, points, 3)
    # label B, 1) uint8
    # seg (B, points, 3) uint8
    return (data, label, seg)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)