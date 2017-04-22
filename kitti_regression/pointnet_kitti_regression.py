import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def get_transform_K(inputs, is_training, bn_decay=None, K = 3):
    """ Transform Net, input is BxNx1xK gray image
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        
        weights = tf.get_variable('weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    #transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

def get_transform(point_cloud, is_training, bn_decay=None, K = 3):
    """ Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value # tf.shape(point_cloud)[0]
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1) # BxNx3 -> BxNx3x1
    net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training, scope='tconv4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=False, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=False, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [128, 3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    #transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def get_model(point_cloud, input_label, is_training, cat_num, batch_size, num_point, bn_decay=None, bn=False):
    """ ConvNet baseline, input is BxNx3 gray image """
    end_points = {}

    mean_xyz, variance_xyz = tf.nn.moments(point_cloud, axes=[1])
    scal_xyz = tf.sqrt(variance_xyz + 0.0001) 
    print(mean_xyz.get_shape(), variance_xyz.get_shape())
    input_image_norm = tf.subtract(point_cloud, mean_xyz)
    input_image_norm = tf.divide(input_image_norm, scal_xyz)

    #batch_size = point_cloud.get_shape()[0].value
    #batch_size = tf.shape(point_cloud)[0] #point_cloud.get_shape()[0].value

    with tf.variable_scope('transform_net1') as sc:
        K = 3
        transform = get_transform(input_image_norm, is_training, bn_decay, K = 3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)

    print("point_cloud_transformed shape", point_cloud_transformed.get_shape())
    input_image = tf.expand_dims(input_image_norm, -1)
    K=3

    #input_image = tf.expand_dims(input_image_norm, -1)

    #print("point_cloud_transformed shape expanded", input_image.get_shape())

    out1 = tf_util.conv2d(input_image, 64, [1,K], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    out2 = tf_util.conv2d(out1, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    out3 = tf_util.conv2d(out2, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv3', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        K = 128
        transform = get_transform_K(out3, is_training, bn_decay, K)

    end_points['transform'] = transform

    squeezed_out3 = tf.reshape(out3, [batch_size, num_point, 128])
    net_transformed = tf.matmul(squeezed_out3, transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    out4 = tf_util.conv2d(net_transformed, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    out5 = tf_util.conv2d(out4, 2048, [1,1], padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    out_max = tf_util.max_pool2d(out5, [num_point,1], padding='VALID', scope='maxpool')

    #one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    #out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

    # regression network
    net = tf.reshape(out_max, [batch_size, -1])
    net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training, scope='cla/fc1')
    net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training, scope='cla/fc2')
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='cla/dp1')
    net = tf_util.fully_connected(net, 5, activation_fn=None, scope='cla/fc3')
    # Bx4

    # segmentation network
    #one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    #out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

    #expand = tf.tile(out_max, [1, num_point, 1, 1])
    #concat = tf.concat(axis=3, values=[expand, out1, out2, out3, out4, out5])

    #net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
    #                    bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
    #net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
    #net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
    #                    bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
    #net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
    #net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
    #                    bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
    #net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
    #                    bn=False, scope='seg/conv4', weight_decay=weight_decay)

    #net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    # net[0:2] -> A
    # net[2:4] -> B
    # net[4] -> h
    # B, 5 

    x_scale = tf.slice(scal_xyz, [0,0], [-1,1])
    y_scale = tf.slice(scal_xyz, [0,1], [-1,1])

    bbox_Ax = tf.slice(net, [0,0], [-1,1]) # Bx1
    bbox_Ax = tf.multiply(bbox_Ax, x_scale)
    bbox_Ax = tf.add(bbox_Ax,      tf.slice(mean_xyz, [0,0], [-1,1]))

    bbox_Ay = tf.slice(net, [0,1], [-1,1])
    bbox_Ay = tf.multiply(bbox_Ay, y_scale)
    bbox_Ay = tf.add(bbox_Ay,      tf.slice(mean_xyz, [0,1], [-1,1])) 

    bbox_Bx = tf.slice(net, [0,2], [-1,1])
    bbox_Bx = tf.multiply(bbox_Bx, x_scale)
    bbox_Bx = tf.add(bbox_Bx,      tf.slice(mean_xyz, [0,0], [-1,1])) 

    bbox_By = tf.slice(net, [0,3], [-1,1])
    bbox_By = tf.multiply(bbox_By, y_scale)
    bbox_By = tf.add(bbox_By,      tf.slice(mean_xyz, [0,1], [-1,1]))

    bbox_l = tf.slice(net,  [0,4], [-1,1])

    # we have to scale 'l' back, but  it has different scales for x and y
    # l' = l * sqrt(x_scale^2 * cos^2(alpha) + y_scale^2 * sin^2 (alpha))
    # alpha = arctan(x, y ) -> note x y are swapped
    # if alpha =  0    -> car points towards us and only x_scale prevails
    # is alpha =  pi/2 -> car is sideways and only y_scale prevails
    alpha  = atan2(bbox_Bx - bbox_Ax, bbox_By - bbox_Ay)
    bbox_l = tf.multiply(bbox_l, tf.sqrt(tf.square(y_scale) * tf.square(tf.sin(alpha)) + tf.square(x_scale) * tf.square(tf.cos(alpha)) ))

    bbox_A = tf.concat([bbox_Ax, bbox_Ay], 1)
    bbox_B = tf.concat([bbox_Bx, bbox_By], 1)
    return bbox_A, bbox_B, bbox_l, end_points

# https://github.com/tensorflow/tensorflow/issues/6095 
def atan2(y, x, epsilon=1.0e-12):
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y+epsilon, y)

    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle

def get_loss(bbox_A_pred, bbox_B_pred, bbox_l_pred, bbox_A, bbox_B, bbox_l, end_points):
    #per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
    #label_loss = tf.reduce_mean(per_instance_label_loss)

    # bbox_xy_pred is Bx2
    # bbox_wh_pred is Bx2

    print(bbox_A.get_shape(), bbox_A_pred.get_shape())
    bbox_A_loss = tf.reduce_mean(tf.nn.l2_loss(bbox_A - bbox_A_pred))
    print(tf.nn.l2_loss(bbox_A - bbox_A_pred).get_shape())
    bbox_B_loss = tf.reduce_mean(tf.nn.l2_loss(bbox_B - bbox_B_pred))
    bbox_l_loss = tf.reduce_mean(tf.nn.l2_loss(bbox_l - bbox_l_pred))

    AB      = tf.subtract(bbox_B,      bbox_A)
    AB_pred = tf.subtract(bbox_B_pred, bbox_A_pred)

    area_loss  = tf.reduce_mean(tf.nn.l2_loss(tf.multiply(tf.norm(AB, axis=1), bbox_l) - tf.multiply(tf.norm(AB_pred, axis=1), bbox_l_pred)))

    angle      = atan2(tf.slice(AB,      [0,1], [-1, 1]), tf.slice(AB,      [0,0], [-1, 1]))
    angle_pred = atan2(tf.slice(AB_pred, [0,1], [-1, 1]), tf.slice(AB_pred, [0,0], [-1, 1]))
    print("angle", angle)
    print("angle_pred", angle_pred)

    angle_loss = tf.reduce_mean(tf.nn.l2_loss(angle - angle_pred))

    midpoint      = (bbox_A +      bbox_B)      / 2.
    midpoint_pred = (bbox_A_pred + bbox_B_pred) / 2.

    head_loss = tf.nn.l2_loss(midpoint - midpoint_pred)


    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1])) - tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 

    total_loss = bbox_A_loss + bbox_B_loss + bbox_l_loss + mat_diff_loss * 1e-3

    return total_loss, bbox_A_loss, bbox_B_loss, bbox_l_loss
    #return 2 * head_loss + area_loss + 5 * angle_loss, head_loss, area_loss, angle_loss


    # Calculate IoU (Jaccard index)
    #      ____---C
    #  D---        \  l 
    #   \           \ 
    #    \    ____---B        
    #     A---   w  
    #
    # Angle(AD,AB) = 90

