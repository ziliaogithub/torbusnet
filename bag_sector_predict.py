import provider_didi
from keras.models import load_model
import rosbag
import rospy
import argparse
import sensor_msgs.point_cloud2 as pc2
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'didi-competition/tracklets/python'))

from diditracklet import *
import point_utils
from generate_tracklet import *
import re

parser = argparse.ArgumentParser(description='Predicts sector where object is detected in bag.')
parser.add_argument('-i', '--input-bag', default='../release2/Data-points/test/19_f2.bag', type=str, help='input bag to process')
parser.add_argument('-o', '--output-dir', default='./img-out', help='output directory for images')
parser.add_argument('-m', '--model', required = True, help='path to hdf5 model')
parser.add_argument('-cd', '--clip-distance', default=50., type=float, help='Clip distance (needs to be consistent with trained model!)')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU inference')

args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = load_model(args.model)
model.summary()
points_per_ring = model.get_input_shape_at(0)[0][1]
match = re.search(r'lidarnet-cla-rings_(\d+)_(\d+)-.*\.hdf5', args.model)
rings = range(int(match.group(1)), int(match.group(2)))
assert len(rings) == model.get_input_shape_at(0)[0][2]

POINT_LIMIT = 65536
cloud = np.empty((POINT_LIMIT, 5), dtype=np.float32)

for topic, msg, t in rosbag.Bag(args.input_bag).read_messages():
    if topic == '/velodyne_points':
        time_prep_start = time.time()
        timestamp = msg.header.stamp.to_nsec()
        points = 0

        for x, y, z, intensity, ring in pc2.read_points(msg):
            cloud[points] = x, y, z, intensity, ring
            points += 1
        time_prep_generator_end = time.time()
        lidar = DidiTracklet.filter_lidar_rings(cloud[:points], rings, points_per_ring, clip=(0., args.clip_distance))

        # TODO: change the net so that we can feed it the output of filter_lidar_rings directly w/o rearraging the arrays
        lidar_d = np.empty((1, points_per_ring, len(rings)), dtype=np.float32)
        lidar_z = np.empty((1, points_per_ring, len(rings)), dtype=np.float32)
        lidar_i = np.empty((1, points_per_ring, len(rings)), dtype=np.float32)
        for ring in range(len(rings)):
            lidar_d[0, :, ring] = lidar[ring, :, 0]
            lidar_z[0, :, ring] = lidar[ring, :, 1]
            lidar_i[0, :, ring] = lidar[ring, :, 2]
        time_prep_end = time.time()
        class_predictions_by_angle = model.predict([lidar_d, lidar_z, lidar_i], batch_size = 1)
        time_infe_end = time.time()
        print 'Total time: %0.3f ms' % ((time_infe_end - time_prep_start) * 1000.0)
        print ' Generator: %0.3f ms' % ((time_prep_generator_end - time_prep_start) * 1000.0)
        print '      Prep: %0.3f ms' % ((time_prep_end - time_prep_start) * 1000.0)
        print ' Inference: %0.3f ms' % ((time_infe_end - time_prep_end)   * 1000.0)

        print(timestamp)
        print(class_predictions_by_angle)

