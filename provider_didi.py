import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'didi-competition/tracklets/python'))

from diditracklet  import *

def get_tracklets(root, date_drive_filelist, xml_filename="tracklet_labels_trainable.xml", box_scaling=(1.,1.,1.)):
    bags = [line.rstrip() for line in open(date_drive_filelist)]
    tracklet_list = [[root, bag.split('/')[0], bag.split('/')[1]] for bag in bags]
    diditracklets = []
    for root,date,drive in tracklet_list:
        diditracklet = DidiTracklet(root, date, drive, xml_filename=xml_filename, box_scaling=box_scaling)
        diditracklets.append(diditracklet)

    return diditracklets
