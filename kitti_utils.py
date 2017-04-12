import numpy as np
import pykitti
import tracklets
from collections import defaultdict
import cv2


M = 10
MIN_HEIGHT    = -2. # from camera (i.e.  -2-1.65 =  3.65m above floor)
MAX_HEIGHT    =  2. # from camera (i.e.  +2-1.65 =  0.35m below floor)
M_HEIGHT      = (MAX_HEIGHT-MIN_HEIGHT)/M
MIN_X = -40.
MAX_X =  40.
MIN_Z =   5.
MAX_Z =  70.
HEIGHT_F_RES = 0.1 # 0.1m for x,z slicing

C_W = 512
C_H = 64

CAMERA_FEATURES = (375, 1242, 3)
HEIGHT_FEATURES = (int((MAX_Z-MIN_Z)/HEIGHT_F_RES), int((MAX_X-MIN_X)/HEIGHT_F_RES), M+2)
F_VIEW_FEATURES = (C_H, C_W, 3)

class Parser(object):
    basedir       = None
    date          = None
    drive         = None
    kitti_data    = None
    tracklet_data = None

    # lidars is a dict indexed by frame: e.g. lidars[10] = np(N,4)
    lidars = {}

    # boxes is a dict indexed by frame:  e.g. boxes[10] = [box, box, ...] 
    _boxes = None # defaultdict(list)

    def __init__(self, basedir, date, drive):
        self.basedir = basedir
        self.date    = date
        self.drive   = drive

        self.kitti_data = pykitti.raw(basedir, date, drive, range(0,1)) #, range(start_frame, start_frame + total_frames))
        self.tracklet_data = tracklets.parseXML(basedir, date, drive)
        self.kitti_data.load_calib()        # Calibration data are accessible as named tuples

    # return list of frames where tracked objects of type only_with are visible (in image)
    def frames(self, only_with):
        frames = []
        for t in self.tracklet_data:
            if t.objectType in only_with:
                for frame_offset in range(t.firstFrame, t.nFrames):
                    if t.truncs[frame_offset]: #is tracklets.Truncation.IN_IMAGE:
                        frames.append(frame_offset)
            else:
                print("UNTRACKED", t.objectType)
        self._init_boxes(only_with)
        return list(set(frames)) # remove duplicates

    def _read_lidar(self, frame):
        kitti_data = pykitti.raw(self.basedir, self.date, self.drive, range(frame,frame+1)) #, range(start_frame, start_frame + total_frames))
        #kitti_data.load_timestamps()   # Timestamps are parsed into timedelta objects
        kitti_data.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]
        #kitti_data.load_oxts()

        self.lidars[frame] = kitti_data.velo[0]
        return

    # initialize self.boxes with a dict containing frame -> [box, box, ...] 
    def _init_boxes(self, only_with):    
        self._boxes = defaultdict(list)
        for t in self.tracklet_data:
            if t.objectType in only_with:
                for frame_offset in range(t.firstFrame, t.nFrames):
                    #if t.truncs[frame_offset]: #is tracklets.Truncation.IN_IMAGE:
                    h,w,l = t.size
                    trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
                        [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
                        [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
                        [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])
                    yaw = t.rots[frame_offset-t.firstFrame][2]   # other rotations are 0 in all xml files I checked
                    assert np.abs(t.rots[frame_offset-t.firstFrame][:2]).sum() == 0, 'object rotations other than yaw given!'
                    rotMat = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw),  np.cos(yaw), 0.0],
                        [        0.0,          0.0, 1.0]])
                    cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(t.trans[frame_offset-t.firstFrame], (8,1)).T
                    self._boxes[frame_offset].append(cornerPosInVelo)
        return

    # return a top view of the lidar image for frame
    # draw boxes for tracked objects if with_boxes is True
    def top_view(self, frame, with_boxes = True):
        # ranges in lidar coords
        X_RANGE = (  0., 70.)
        Y_RANGE = (-40., 40.)
        Z_RANGE = ( -2.,  2.)
        RES = 0.2

        Y_PIXELS = int((Y_RANGE[1]-Y_RANGE[0]) / RES)
        X_PIXELS = int((X_RANGE[1]-X_RANGE[0]) / RES)

        top_view = np.zeros(shape=(X_PIXELS, Y_PIXELS, 3),dtype=np.float32)

        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars

        
          # convert from lidar x y to top view X Y 
        def toY(y):
            return int((y-Y_RANGE[0]) // RES)
        def toX(x):
            return int((X_PIXELS-1) - (x-X_RANGE[0]) // RES)
        def toXY(x,y):
            return (toY(y), toX(x))

        lidar  = self.lidars[frame]
        in_img, outside_img = self.__project(lidar, return_projection=False, return_velo_in_img=True, return_velo_outside_img=True)

        for point in in_img:
            x, y = point[0], point[1]
            if (x >= X_RANGE[0]) and (x <= X_RANGE[1]) and (y >= Y_RANGE[0]) and (y <= Y_RANGE[1]):
                top_view[toXY(x,y)[::-1] ] = (1., 1., 1.)

        for point in outside_img:
            x, y = point[0], point[1]
            if (x >= X_RANGE[0]) and (x <= X_RANGE[1]) and (y >= Y_RANGE[0]) and (y <= Y_RANGE[1]):
                top_view[toXY(x,y)[::-1] ] = (0.5, 0.5, 0.5)

        if with_boxes:
            assert self._boxes is not None
            for box in self._boxes[frame]:
                # bounding box in image coords (x,y) defined by a->b->c->d
                a = np.array([toXY(box[0,0], box[1,0])])
                b = np.array([toXY(box[0,1], box[1,1])])
                c = np.array([toXY(box[0,2], box[1,2])])
                d = np.array([toXY(box[0,3], box[1,3])])

                cv2.polylines(top_view, [np.int32((a,b,c,d)).reshape((-1,1,2))], True,(1.,0.,0.), thickness=1)

                for point in self._lidar_in_box(frame, box):
                    x = point[0]
                    y = point[1]
                    top_view[toXY(x,y)[::-1] ] = (0., 1., 1.)

        return top_view

    # returns lidar points that are inside a given box
    def _lidar_in_box(self, frame, box):

        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        
        lidar = self.lidars[frame]
        p = lidar[:,0:3]

        # determine if points in M are inside a rectangle defined by AB AD (AB and AD are orthogonal)
        # tdlr: they are iff (0<AM⋅AB<AB⋅AB)∧(0<AM⋅AD<AD⋅AD)
        # http://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
        a    = np.array([box[0,0], box[1,0]])
        b    = np.array([box[0,1], box[1,1]])
        d    = np.array([box[0,3], box[1,3]])
        ab   = b-a
        ad   = d-a
        abab = np.dot(ab,ab)
        adad = np.dot(ad,ad)

        amab = np.squeeze(np.dot(np.array([p[:,0]-a[0], p[:,1]-a[1]]).T, ab.reshape(-1,2).T))
        amad = np.squeeze(np.dot(np.array([p[:,0]-a[0], p[:,1]-a[1]]).T, ad.reshape(-1,2).T))
        h    = np.array([box[2,0], box[2,4]])

        in_box_idx = np.where((abab >= amab) & (amab >= 0.) & (amad >= 0.) & (adad >= amad) & (p[:,2] >= h[0]) & (p[:,2] <= h[1]))
        points_in_box = np.squeeze(lidar[in_box_idx,:])

        return points_in_box


    # given array of points with shape (N_points) and projection matrix w/ shape (3,4)
    # projects points onto a 2d plane
    # returns projected points (N_F_points,2) and 
    # their LIDAR counterparts (N_F_points,3) (unless return_velo_in_img is set to False)
    #
    # N_F_points is the total number of resulting points after filtering (only_forward Z>0 by default)
    # and optionally filtering points projected into the image dimensions spec'd by dim_limit:
    # 
    # Optionally providing dim_limit (sx,sy) limits projections that end up within (0-sx,0-sy)
    # only_forward to only get points with Z >= 0
    #
    def __project(self, points, dim_limit=(1242, 375), 
        only_forward = True, 
        return_projection = True, 
        return_velo_in_img = True, 
        return_velo_outside_img = False, 
        return_append = None):

        assert return_projection or return_velo_in_img

        K = self.kitti_data.calib.K_cam2
        R = np.eye(4)
        R[0:3,0:3] = K
        T= np.dot(R, self.kitti_data.calib.T_cam2_velo)[0:3]
        px = points

        if only_forward:
            only_forward_filter = px[:,0] >= 0.
            px = px[only_forward_filter]
        if points.shape[1] < T.shape[1]:
            px = np.concatenate((px,np.ones(px.shape[0]).reshape(-1,1)), axis=1)
        projection = np.dot(T, px.T).T

        norm = np.dot(projection[:,T.shape[0]-1].reshape(-1,1), np.ones((1,T.shape[0]-1)))
        projection = projection[:,0:T.shape[0]-1] / norm

        if dim_limit is not None:
            x_limit, y_limit = dim_limit[0], dim_limit[1]
            only_in_img = (projection[:,0] >= 0.) & (projection[:,0] < x_limit) & (projection[:,1] >= 0.) & (projection[:,1] < y_limit)
            projection = projection[only_in_img]
            if return_velo_in_img:
                if return_velo_outside_img:
                    _px = px[~ only_in_img]
                px = px[only_in_img]
        if return_append is not None:
            appended = return_append[only_forward_filter][only_in_img]
            assert return_projection and return_velo_in_img
            return (projection, np.concatenate((px[:,0:3], appended.reshape(-1,1)), axis=1).T)
        if return_projection and return_velo_in_img:
            return (projection, px)
        elif (return_projection is False) and (return_velo_in_img):
            if return_velo_outside_img:
                return px, _px
            else:
                return px
        return projection

    def build_height_features(self, point_cam_in_img):
        assert False #function not tested

        height_features     = np.zeros(shape=(int((MAX_Z-MIN_Z)/HEIGHT_F_RES), int((MAX_X-MIN_X)/HEIGHT_F_RES), M+2),dtype=np.float32)
        max_height_per_cell = np.zeros_like(height_features[:,:,1])
        for p in point_cam_in_img.T:
            x = p[0] 
            y = MAX_HEIGHT-np.clip(p[1], MIN_HEIGHT, MAX_HEIGHT)
            z = p[2]
            if (x >= MIN_X) and (x < MAX_X) and (z >= MIN_Z) and (z < MAX_Z): 
                m  = int(y//M_HEIGHT) 
                xi = int((x+MIN_X)//HEIGHT_F_RES)
                zi = int((z-MIN_Z)//HEIGHT_F_RES)
                height_features[zi, xi, m] = max(y, height_features[zi, xi, m])
                if y >= max_height_per_cell[zi, xi]:
                    max_height_per_cell[zi, xi] = y
                    height_features[zi, xi, M]  = p[3] # intensity
                height_features[zi, xi, M+1] += 1
        log64 = np.log(64)
        height_features[:, :, M+1] = np.clip(np.log(1+height_features[:, :, M+1])/log64, 0.,1.)
        return height_features

def build_front_view_features(point_cam_in_img):
    delta_theta = 0.08 / (180./np.pi) # horizontal resolution
    delta_phi   = 0.4  / (180./np.pi) # vertical resolution as per http://velodynelidar.com/hdl-64e.html

    c_projection = np.empty((point_cam_in_img.shape[1],5)) # -> c,r,height,distance,intensity
    points = point_cam_in_img.T
    # y in lidar is [0] in cam (x)
    # x in lidar is [2] in cam (z)
    # z in lidar is [1] in cam (y)
    c_range = (-40  /(180./np.pi), 40   /(180./np.pi))
    r_range = (-2.8 /(180./np.pi), 15.3 /(180./np.pi))

    c_projection[:,0] = np.clip(
        np.arctan2(points[:,0], points[:,2]), 
        c_range[0], c_range[1]) # c
    c_projection[:,1] = np.clip(
        np.arctan2(points[:,1], np.sqrt(points[:,2] **2 + points[:,0] **2 )), 
        r_range[0], r_range[1]) # r
    c_projection[:,2] = MAX_HEIGHT-np.clip(points[:,1], MIN_HEIGHT, MAX_HEIGHT)       # height
    c_projection[:,3] = np.sqrt(points[:,0] **2 + points[:,1] **2 + points[:,2] **2 ) # distance
    c_projection[:,4] = points[:,3]

    c_norm = np.zeros((C_H,C_W,3))
    c_norm[np.int32((C_H - 1) * (c_projection[:,1] - r_range[0]) // (r_range[1]-r_range[0])), np.int32((C_W - 1) * (c_projection[:,0] - c_range[0]) // (c_range[1]-c_range[0]))] = c_projection[:,2:5]#.reshape(-1,3)

    return c_norm




"""Miscellaneous utility functions."""
from functools import reduce


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
