import numpy as np
import os
import sys
PYKITTI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pykitti/")
sys.path.append(PYKITTI_DIR)
sys.path.append(os.path.dirname(PYKITTI_DIR))
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
    kitti_cat_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Sitter', 'Cyclist', 'Tram', 'Misc', 'Person (sitting)']
    kitti_cat_idxs  = range(1,1+len(kitti_cat_names))

    LIDAR_ANGLE = np.pi/6.

    def __init__(self, basedir, date, drive):
        self.basedir = basedir
        self.date    = date
        self.drive   = drive

        self.kitti_data = pykitti.raw(basedir, date, drive, range(0,1)) #, range(start_frame, start_frame + total_frames))
        self.tracklet_data = tracklets.parseXML(basedir, date, drive)
        self.kitti_data.load_calib()        # Calibration data are accessible as named tuples

        # lidars is a dict indexed by frame: e.g. lidars[10] = np(N,4)
        self.lidars = {}

        # images is a dict indexed by frame: e.g. lidars[10] = np(SY,SX,3)
        self.images = {}
        self.im_dim = (1242, 375) # by default

        # boxes is a dict indexed by frame:  e.g. boxes[10] = [box, box, ...] 
        self._boxes = None # defaultdict(list)

    # return list of frames where tracked objects of type only_with are visible (in image)
    def frames(self, only_with):
        frames = []
        for t in self.tracklet_data:
            if t.objectType in only_with:
                for frame_offset in range(t.firstFrame,t.firstFrame + t.nFrames):
                    if t.truncs[frame_offset-t.firstFrame]: #is tracklets.Truncation.IN_IMAGE:
                        frames.append(frame_offset)
                    else:
                        print("No truncation data!")
            else:
                print("UNTRACKED", t.objectType)
        self._init_boxes(only_with)
        return list(set(frames)) # remove duplicates

    def _read_lidar(self, frame):
        if frame not in self.kitti_data.frame_range:
            self.kitti_data = pykitti.raw(self.basedir, self.date, self.drive, range(frame,frame+1)) #, range(start_frame, start_frame + total_frames))
            self.kitti_data.load_calib() 
        assert frame in self.kitti_data.frame_range
        self.kitti_data.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]
        self.lidars[frame] = self.kitti_data.velo[0]
        return

    def _read_image(self, frame):
        if frame not in self.kitti_data.frame_range:
            self.kitti_data = pykitti.raw(self.basedir, self.date, self.drive, range(frame,frame+1)) #, range(start_frame, start_frame + total_frames))
            self.kitti_data.load_calib() 
        assert frame in self.kitti_data.frame_range
        self.kitti_data.load_rgb()
        self.images[frame] = self.kitti_data.rgb[0].left

        (sx,sy) = self.images[frame].shape[::-1][1:]

        if self.im_dim != (sx,sy):
            print("WARNING changing default dimensions to", (sx,sy))
            self.im_dim = (sx,sy)

        return

    # initialize self.boxes with a dict containing frame -> [box, box, ...] 
    def _init_boxes(self, only_with):
        assert self._boxes is None
        self._boxes = defaultdict(list)
        for t in self.tracklet_data:
            if t.objectType in only_with:
                for frame_offset in range(t.firstFrame, t.firstFrame + t.nFrames):
                    if t.truncs[frame_offset-t.firstFrame] == tracklets.Truncation.IN_IMAGE:
                        h,w,l = t.size
                        # in velo:
                        # A       D
                        #
                        # B       C
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

    # given lidar points, subsample POINTS by removing points from voxels with highest density
    def _lidar_subsample(self, lidar, POINTS):
        X_RANGE = (  0., 70.)
        Y_RANGE = (-40., 40.)
        Z_RANGE = ( -2.,  2.)
        RES = 0.2

        NX = 10
        NY = 10
        NZ = 4

        bins, edges = np.histogramdd(lidar[:,0:3], bins = (NX, NY, NZ))

        bin_target = np.array(bins, dtype=np.int32)
        # inefficient but effective, TODO optimize (easy)
        for i in range(np.sum(bin_target) - POINTS):
            bin_target[np.unravel_index(bin_target.argmax(), bin_target.shape)] -= 1

        target_n = np.sum(bin_target)
        assert target_n >= POINTS

        subsampled = np.empty_like(lidar[:target_n,:])

        i = 0
        nx,ny,nz = bin_target.shape
        for (x,y,z),v in np.ndenumerate(bin_target):
            if v > 0:
                XX = edges[0][x:x+2]
                YY = edges[1][y:y+2]
                ZZ = edges[2][z:z+2]
                # edge cases needed b/c histogramdd includes righest-most edge in bin
                if x < (nx-1):
                    sublidar = lidar[(lidar[:,0] >= XX[0]) & (lidar[:,0] <  XX[1])]
                else:
                    sublidar = lidar[(lidar[:,0] >= XX[0]) & (lidar[:,0] <= XX[1])]
                if y < (ny-1):
                    sublidar = sublidar[(sublidar[:,1] >= YY[0]) & (sublidar[:,1] <  YY[1])]
                else:
                    sublidar = sublidar[(sublidar[:,1] >= YY[0]) & (sublidar[:,1] <= YY[1])]
                if z < (nz-1):
                    sublidar = sublidar[(sublidar[:,2] >= ZZ[0]) & (sublidar[:,2] <  ZZ[1])]
                else:
                    sublidar = sublidar[(sublidar[:,2] >= ZZ[0]) & (sublidar[:,2] <= ZZ[1])]
                assert sublidar.shape[0] == bins[x,y,z]
                assert sublidar.shape[0] >= v
                subsampled[i:(i+v)] = sublidar[np.random.choice(range(sublidar.shape[0]), v, replace=False)]
                i += v
        return subsampled

    # For each detected object in frame, returns a list of 
    # - lidar_in_2d_bbox (MAX_POINTS, 3)
    # - label of object
    # - bbox 6 -> (Ax, Ay, Bx, By, l, h) 
    def lidar_cone_of_detected_objects(self, frame, return_image=False):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]

        if return_image:
            if frame not in self.images:
                self._read_image(frame)
                assert frame in self.images
            image = self.images[frame]

        assert self._boxes is not None

        lidar_cones = []
        for box in self._boxes[frame]:
            lidar_in_2d_bbox = self.__lidar_in_2d_box(lidar, box)

            # we'll pass two lower points A and B, and l and h (that defines the bounding box in 3d)
            A = np.array([box[0,0], box[1,0]])
            B = np.array([box[0,1], box[1,1]])
            C = np.array([box[0,3], box[1,3]])
            l = np.linalg.norm(C-B)
            h = np.array([box[2,4]-box[2,0]])
            #import code
            #code.interact(local=locals())
            bbox = np.concatenate((A,B,[l],h), axis=0)
            lidar_cones.append((lidar_in_2d_bbox, bbox, Parser.kitti_cat_idxs[0])) # TODO: deal with categories 

            if return_image:
                bbox_l, bbox_h = self.__box_to_2d_box(box)
                _bbox_l = (int(bbox_l[0]), int(bbox_l[1]))
                _bbox_h = (int(bbox_h[0]), int(bbox_h[1]))
                image = cv2.rectangle(image, _bbox_l, _bbox_h, (1.,1.,1.))
                print(lidar_in_2d_bbox.shape[0], "found in", bbox_l, bbox_h)

        if return_image:
            top_view = self.top_view(frame, with_boxes = True, SX = image.shape[1])
            print("top_view shape", top_view.shape)
            image = np.concatenate((image, top_view), axis=0)
            print("Frame", frame, "Image now", image.shape)

        if return_image:
            return lidar_cones, image
        return lidar_cones


    def lidar_with_label(self, frame, MAX_POINTS=None):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]
        lidar, lidar_o = self.__project(lidar, return_projection=False, return_velo_in_img=True, return_velo_outside_img=True)
        if self.LIDAR_ANGLE is not None:
            lidar_o = lidar_o[np.arctan2(lidar_o[:,0],np.absolute(lidar_o[:,1])) >= self.LIDAR_ANGLE]
        lidar = np.concatenate((lidar, lidar_o), axis=0)

        if (MAX_POINTS is not None) and (MAX_POINTS < lidar.shape[0]):
            lidar = self._lidar_subsample(lidar, MAX_POINTS)

        _lidar_with_label = np.empty((lidar.shape[0], 5))
        _lidar_with_label[:,0:4] = lidar[:,:]
        _lidar_with_label[:,4] = 0
        
        assert self._boxes is not None
        for box in self._boxes[frame]:
            idx = self.__lidar_in_box(_lidar_with_label, box, return_idx_only = True)
            _lidar_with_label[idx,4] = 1

        return _lidar_with_label



    # returns lidar points with associated label into (Nx5 array)
    # defaults to MAX_POINTS if not provided
    def lidar_with_label(self, frame, MAX_POINTS=None):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]
        lidar, lidar_o = self.__project(lidar, return_projection=False, return_velo_in_img=True, return_velo_outside_img=True)
        if self.LIDAR_ANGLE is not None:
            lidar_o = lidar_o[np.arctan2(lidar_o[:,0],np.absolute(lidar_o[:,1])) >= self.LIDAR_ANGLE]
        lidar = np.concatenate((lidar, lidar_o), axis=0)

        if (MAX_POINTS is not None) and (MAX_POINTS < lidar.shape[0]):
            lidar = self._lidar_subsample(lidar, MAX_POINTS)

        _lidar_with_label = np.empty((lidar.shape[0], 5))
        _lidar_with_label[:,0:4] = lidar[:,:]
        _lidar_with_label[:,4] = 0
        
        assert self._boxes is not None
        for box in self._boxes[frame]:
            idx = self.__lidar_in_box(_lidar_with_label, box, return_idx_only = True)
            _lidar_with_label[idx,4] = 1

        return _lidar_with_label

    # return a top view of the lidar image for frame
    # draw boxes for tracked objects if with_boxes is True
    def top_view(self, frame, with_boxes = True, lidar_override = None, SX = None):
        # ranges in lidar coords, including lower end, excluding higher end
        X_RANGE = (  0., 100.) 
        Y_RANGE = (-40.,  40.)
        Z_RANGE = ( -2.,   2.)
        if SX is None:
            RES = 0.2
            Y_PIXELS = int((Y_RANGE[1]-Y_RANGE[0]) / RES)
        else:
            Y_PIXELS = SX
            RES = (Y_RANGE[1]-Y_RANGE[0]) / SX
        import math
        X_PIXELS = int(math.ceil((X_RANGE[1]-X_RANGE[0]) / RES))
        print(X_PIXELS, Y_PIXELS, RES)

        top_view = np.zeros(shape=(X_PIXELS, Y_PIXELS, 3),dtype=np.float32)

        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars

        # convert from lidar x y to top view X Y 
        def toY(y):
#            return int((Y_RANGE[1]-y) // RES)
            return int((Y_PIXELS-1) - (y-Y_RANGE[0]) // RES)
        def toX(x):
            return int((X_PIXELS-1) - (x-X_RANGE[0]) // RES)
        def toXY(x,y):
            return (toY(y), toX(x))

        def inRange(x,y):
            return (x >= X_RANGE[0]) and (x < X_RANGE[1]) and (y >= Y_RANGE[0]) and (y < Y_RANGE[1])

        if lidar_override is not None:
            lidar = lidar_override
        else:
            lidar = self.lidars[frame]

        in_img, outside_img = self.__project(lidar, return_projection=False, return_velo_in_img=True, return_velo_outside_img=True)

        for point in in_img:
            x, y = point[0], point[1]
            if inRange(x,y):
                top_view[toXY(x,y)[::-1] ] = (1., 1., 1.)

        for point in outside_img:
            x, y = point[0], point[1]
            if inRange(x,y):
                c = (0.2,0.2,0.2)
                if (self.LIDAR_ANGLE is not None) and (np.arctan2(x,np.absolute(y)) >= self.LIDAR_ANGLE):
                    c = (0.5, 0.5, 0.5)    
                top_view[toXY(x,y)[::-1] ] = c

        if with_boxes:
            assert self._boxes is not None
            for box in self._boxes[frame]:
                # bounding box in image coords (x,y) defined by a->b->c->d
                a = np.array([toXY(box[0,0], box[1,0])])
                b = np.array([toXY(box[0,1], box[1,1])])
                c = np.array([toXY(box[0,2], box[1,2])])
                d = np.array([toXY(box[0,3], box[1,3])])

                cv2.polylines(top_view, [np.int32((a,b,c,d)).reshape((-1,1,2))], True,(1.,0.,0.), thickness=1)

                lidar_in_box = self._lidar_in_box(frame, box)
                print(lidar_in_box.shape)
                for point in lidar_in_box:
                    x,y = point[0], point[1]
                    if inRange(x,y):
                        top_view[toXY(x,y)[::-1] ] = (0., 1., 1.)

        return top_view

    def __box_to_2d_box(self, box):
        print("Frame", self.kitti_data.frame_range, "box", box)
        box_in_img = self.__project(box.T, return_projection=True, dim_limit=None, return_velo_in_img=False, return_velo_outside_img=False)
        # some boxes are behind the viewpoint (eg. frame 70 @ drive 0036 ) and would return empty set of points
        # so we return an empty box
        print("box_in_img.shape", box_in_img.shape)
        if box_in_img.shape[0] != 8:
            return (0,0),(0,0)
        #print("lidar box", box.T,"in img", box_in_img)
        dim_limit = self.im_dim
        print(dim_limit, box_in_img)
        # clip 2d box corners within image
        box_in_img[:,0] = np.clip(box_in_img[:,0], 0, dim_limit[0])
        box_in_img[:,1] = np.clip(box_in_img[:,1], 0, dim_limit[1])
        # get 2d bbox
        bbox_l =(np.amin(box_in_img[:,0]), np.amin(box_in_img[:,1]))
        bbox_h =(np.amax(box_in_img[:,0]), np.amax(box_in_img[:,1]))
        return bbox_l, bbox_h


    def __lidar_in_2d_box(self, lidar, box):
        bbox_l, bbox_h = self.__box_to_2d_box(box)
        #print("2d clipping box", bbox_l, bbox_h, "filtering", lidar.shape)
        lidar_in_2d_box = self.__project(lidar, 
            return_projection=False, dim_limit=bbox_h, dim_limit_zero=bbox_l, return_velo_in_img=True, return_velo_outside_img=False)
        #print("got", lidar_in_2d_box.shape, "in box")
        return lidar_in_2d_box

    # returns lidar points that are inside a given box, or just the indexes 
    def _lidar_in_box(self, frame, box):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        
        lidar = self.lidars[frame]
        return self.__lidar_in_box(lidar, box)

    # returns lidar points that are inside a given box, or just the indexes 
    def __lidar_in_box(self, lidar, box, return_idx_only = False):

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
        if return_idx_only:
            return in_box_idx

        points_in_box = np.squeeze(lidar[in_box_idx,:], axis=0)
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
    def __project(self, points, 
        dim_limit=(-1,-1), 
        dim_limit_zero=(0,0),
        only_forward = True, 
        return_projection = True, 
        return_velo_in_img = True, 
        return_velo_outside_img = False, 
        return_append = None):

        if dim_limit == (-1,-1):
            dim_limit = self.im_dim

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
            x_limit_z, y_limit_z = dim_limit_zero[0], dim_limit_zero[1]
            only_in_img = (projection[:,0] >= x_limit_z) & (projection[:,0] < x_limit) & (projection[:,1] >= y_limit_z) & (projection[:,1] < y_limit)
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
