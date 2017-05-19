import provider_didi
from keras.models import load_model
import rosbag
import rospy
import argparse
import sensor_msgs.point_cloud2 as pc2
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'didi-competition/tracklets/python'))

from diditracklet import *
import point_utils

parser = argparse.ArgumentParser(description='Predict position of obstacle and write tracklet xml file.')
parser.add_argument('-i', '--input-bag', type=str, required=True, help='input bag to process')
parser.add_argument('-o', '--output-xml', type=str, required=True, help='output xml')
parser.add_argument('-m', '--model', help='path to hdf5 model')

args = parser.parse_args()

if args.model:
    model = load_model(args.model)

POINT_LIMIT = 65536
cloud = np.empty((POINT_LIMIT, 4), dtype=np.float32)

for topic, msg, t in rosbag.Bag(args.input_bag).read_messages():
    if topic == '/velodyne_points':
        timestamp = msg.header.stamp.to_nsec()
        points = 0
        for x, y, z, intensity, ring in pc2.read_points(msg):
            cloud[points] = x, y, z, intensity
            points += 1
        lidar = DidiTracklet.filter_lidar(cloud[:points],  num_points = 27000, remove_capture_vehicle=True, max_distance = 25)
        print(timestamp)
        print(points)
        print(cloud.shape)
        print(lidar.shape)

