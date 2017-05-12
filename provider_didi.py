__author__ = 'aureliabustos'

import os
import sys
import numpy as np




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'didi-competition/tracklets/python'))

from diditracklet  import *

# get a list (in the form: date/drive/frame_id) of all frames available in list_filename
# example of the path to traverse: /Volumes/cine/release2/Data-points-processed/1/18/lidar/frame.npy
def getDataFiles(list_dirname):

    frame_list = []
    for (_, dirnames, _) in os.walk(list_dirname):
        for date in dirnames:
            for (_, drivenames, _) in os.walk(list_dirname + "/" + date):
                for drive in drivenames:
                    for (_, _, files)  in os.walk(list_dirname + "/" + date + "/" + drive + "/lidar"):
                        files = sorted(files)
                        for f in files:
                            f = f.strip(".npy")
                            frame_list.append(date + "/" + drive + "/" + f)

    return frame_list



def get_tracklets(directory):
    root = directory
    trackle_list =  [[root,"1","10"], [root,"1","14_f"], [root,"1","11"], [root,"1","3"]]#,[root,"1","13"],[root,"1","19"]] #TODO: read from csv
    diditracklets = []
    for root,date,drive in trackle_list:
        diditracklet = DidiTracklet(root, date, drive)
        diditracklets.append(diditracklet)

    return diditracklets

def loadDataFile(diditracklet,  num_points, num_frames=64):
    """

    :param diditracklet: an object of class Diditracklet
    :param num_points:  the num of lidar points to get for each frame
    :param num_frames:  the num of  frames to get for each tracklet
    :return: the data and label for training the model. Data is an array of frames with its lidar points. Shape [n frames, 3]
    label is an array of frames with the centroids of the objects to detect (only x and y). Shape [n frames, 2]
    """
    data = []
    label = []
    frames =  diditracklet.frames()
    for frame in frames:
        data.append(diditracklet.get_lidar(frame, num_points)[:,:4])
        label.append(list(diditracklet.get_box_centroid(frame)[:2]))
    return (np.array(data), np.array(label))

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
    d =  data[idx, ...]
    l =  labels[idx, ...]
    return d, l, idx