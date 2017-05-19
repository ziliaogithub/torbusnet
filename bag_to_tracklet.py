import rosbag
import rospy
import argparse
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
parser.add_argument('-i', '--ibag', type=str, nargs='?', required=True, help='input bag to process')
parser.add_argument('-o', '--obag', type=str, default ='/dev/null', nargs='?', help='output bag')
parser.add_argument('-t', '--topics', type=str, nargs='+', default='/velodyne/points, camera', help='topics to filter')
parser.add_argument('-s', '--seconds', type=float, default =0.1, nargs='?', help='time threshold in seconds, default: 0.1')
args = parser.parse_args()
filter_topics = args.topics
seconds       = args.seconds
obag          = args.obag
ibag          = args.ibag



def predict(frame_points = None):
    obs_poses = np.array(3)   #(x,y,z)
    obs_dim = np.array(3)     #(h,l,w)
    #predict function
    return  obs_poses,obs_dim


def interpolate_by_camera_time(camera_stamp = None, frame_obs_collection = None):
    return  frame_obs_collection[-1] #return the most recent one assuming they are ordered by frame_stamps


def obs_dim_mode(frame_obs_collection = None):
    m = stats.mode(frame_obs_collection)
    return m


frame_obs_collection = [] # [frame_stamp: (x,y,z),(h,l,w)]
with rosbag.Bag(obag, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(ibag).read_messages():
        newstamp = msg.header.stamp if msg._has_header else t
        if topic in filter_topics:  # read frames from topic /velodyne/points
            if msg._has_header:
                frame_stamp = msg.header.stamp
                #read velodyne points
                if msg.header.seq == "camera":
                    interpolated_frame_obs = interpolate_by_camera_time(frame_stamp, frame_obs_collection)
                    #create tracklet  #TODO: crear un tracklet y ponerle la "pose" en cada frame de la camara y la moda de las dimensiones
                else:
                    frame_points = None #TODO
                    obs_poses,obs_dim = predict(frame_points)
                    frame_obs_collection.append({frame_stamp: [obs_poses,obs_dim]})





