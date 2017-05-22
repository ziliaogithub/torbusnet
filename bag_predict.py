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

parser = argparse.ArgumentParser(description='Predict position of obstacle and write tracklet xml file.')
parser.add_argument('-i', '--input-bag', type=str, required=True, help='input bag to process')
parser.add_argument('-o', '--output-xml-filename', type=str, required=True, help='output xml')
parser.add_argument('-m', '--model', help='path to hdf5 model')
parser.add_argument('-l', '--lidar', action='store_true', help='sync frames to lidar instead of camera')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')

args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.model:
    model = load_model(args.model)
    model.summary()

POINT_LIMIT = 65536
cloud = np.empty((POINT_LIMIT, 4), dtype=np.float32)

collection = TrackletCollection()
obs_tracklet = Tracklet(object_type='Car', l=0., w=0., h=0., first_frame=0)

def get_pose(last_t):
    pose = {}
    pose['tx'] = last_t[0]
    pose['ty'] = last_t[1]
    pose['tz'] = last_t[2]
    pose['rx'] = 0.
    pose['ry'] = 0.
    pose['rz'] = 0.
    pose['status'] = 1
    return pose

last_t = np.zeros(3)
for topic, msg, t in rosbag.Bag(args.input_bag).read_messages():
    if topic == '/velodyne_points':
        time_prep_start = time.time()
        timestamp = msg.header.stamp.to_nsec()
        points = 0
#        xcloud = np.empty((msg.width, 5), dtype=np.float32)
#        print(xcloud.__dict__)
#        xcloud._numpy_ndarray__internals__['data'] = msg.data
#        xcloud.__internals__.strides = (22,4)
        fmt = '<fffxxxxfH'
        for x, y, z, intensity, ring in pc2.read_points(msg):
            cloud[points] = x, y, z, intensity
            points += 1
        time_prep_generator_end = time.time()
        lidar = DidiTracklet.filter_lidar(cloud[:points],  num_points = 24000, remove_capture_vehicle=True, max_distance = 25, print_time = True)
        time_prep_end = time.time()
        last_t, last_s = model.predict(np.expand_dims(lidar, axis=0), batch_size = 1)
        time_infe_end = time.time()
        print 'Total time: %0.3f ms' % ((time_infe_end - time_prep_start) * 1000.0)
        print ' Generator: %0.3f ms' % ((time_prep_generator_end - time_prep_start) * 1000.0)
        print '      Prep: %0.3f ms' % ((time_prep_end - time_prep_start) * 1000.0)
        print ' Inference: %0.3f ms' % ((time_infe_end - time_prep_end)   * 1000.0)

        last_t = np.squeeze(last_t, axis=0)
        last_s = np.squeeze(last_s, axis=0)

        print(last_t, last_s)
        print(timestamp)
        print(points)
        print(lidar.shape)

        if args.lidar:
            print("Saving pose (lidar):", last_t)
            obs_tracklet.poses.append(get_pose(last_t))

    if topic == '/image_raw':
        if not args.lidar:
            print("Saving pose (camera):", last_t)
            obs_tracklet.poses.append(get_pose(last_t))

obs_tracklet.h = last_s[0]
obs_tracklet.w = last_s[1]
obs_tracklet.l = last_s[2]
collection.tracklets.append(obs_tracklet)

tracklet_path = os.path.join('.', args.output_xml_filename)
collection.write_xml(tracklet_path)





