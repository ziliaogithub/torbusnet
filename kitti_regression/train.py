import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import pointnet_kitti_regression as model

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()

import kitti_utils
import kitti_provider_regression as provider

hdf5_data_dir = os.path.join(BASE_DIR, '../../kitti/')

# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# todo: include NONE category
all_cats = [[kitti_cat_name, kitti_cat_idx] for (kitti_cat_name, kitti_cat_idx) in zip (kitti_utils.Parser.kitti_cat_names, kitti_utils.Parser.kitti_cat_idxs)]
NUM_CATEGORIES = 2 # len(all_cats)

print("NUM_CATEGORIES",NUM_CATEGORIES)
print("all_cats",all_cats)

print('#### Batch Size: ', batch_size)
print('#### Point Number: ', point_num)
print('#### Training using GPU: %d' % FLAGS.gpu)

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: ', TRAINING_EPOCHES)

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_tracklets.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'validate_tracklets.txt')

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3), name="pointclouds")
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES), name="input_label")
    bbox_A_ph = tf.placeholder(tf.float32, shape=(batch_size,2), name="bbox_A")
    bbox_B_ph = tf.placeholder(tf.float32, shape=(batch_size,2), name="bbox_B")
    bbox_l_ph = tf.placeholder(tf.float32, shape=(batch_size,1), name="bbox_l")

    #seg_ph = tf.placeholder(tf.int32, shape=(batch_size, point_num))
    return pointclouds_ph, input_label_ph, bbox_A_ph, bbox_B_ph, bbox_l_ph

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_ph, input_label_ph, bbox_A_ph, bbox_B_ph, bbox_l_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0, trainable=False)
            '''
            learning_rate = tf.train.exponential_decay(
                            BASE_LEARNING_RATE,     # base learning rate
                            batch * batch_size,     # global_var indicating the number of steps
                            DECAY_STEP,             # step size
                            DECAY_RATE,             # decay rate
                            staircase=True          # Stair-case or continuous decreasing
                            )
            '''
            learning_rate = 1e-3 # tf.maximum(learning_rate, LEARNING_RATE_CLIP)
        
            bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)
            batch_op = tf.summary.scalar('batch_number', batch)
            bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)
 
            bbox_A_pred, bbox_B_pred, bbox_l_pred, end_points = model.get_model(pointclouds_ph, input_label_ph, \
                    is_training=is_training_ph, bn_decay=bn_decay, cat_num=NUM_CATEGORIES, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)

            # model.py defines both classification net and segmentation net, which share the common global feature extractor network.
            # In model.get_loss, we define the total loss to be weighted sum of the classification and segmentation losses.
            # Here, we only train for segmentation network. Thus, we set weight to be 1.0.
            loss, bbox_A_loss, bbox_B_loss, bbox_l_loss = model.get_loss(bbox_A_pred, bbox_B_pred, bbox_l_pred, bbox_A_ph, bbox_B_ph, bbox_l_ph, end_points)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            bbox_A_training_loss_ph = tf.placeholder(tf.float32, shape=())
            bbox_A_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            bbox_B_training_loss_ph = tf.placeholder(tf.float32, shape=())
            bbox_B_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            bbox_l_training_loss_ph = tf.placeholder(tf.float32, shape=())
            bbox_l_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

            bbox_A_training_loss_sum_op = tf.summary.scalar('bbox_A_training_loss', bbox_A_training_loss_ph)
            bbox_A_testing_loss_sum_op = tf.summary.scalar('bbox_A_testing_loss', bbox_A_testing_loss_ph)

            bbox_B_training_loss_sum_op = tf.summary.scalar('bbox_B_training_loss', bbox_B_training_loss_ph)
            bbox_B_testing_loss_sum_op = tf.summary.scalar('bbox_B_testing_loss', bbox_B_testing_loss_ph)

            bbox_l_training_loss_sum_op = tf.summary.scalar('bbox_l_training_loss', bbox_l_training_loss_ph)
            bbox_l_testing_loss_sum_op = tf.summary.scalar('bbox_l_testing_loss', bbox_l_testing_loss_ph)

            train_variables = tf.trainable_variables()

            trainer = tf.train.AdamOptimizer(learning_rate)
            train_op = trainer.minimize(loss, var_list = train_variables, global_step = batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
        print("train_file_list", train_file_list)
        num_train_file = len(train_file_list)
        test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
        num_test_file = len(test_file_list)

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        def train_one_epoch(train_file_idx, epoch_num):
            is_training = True

            for i in range(num_train_file):
                cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[i]])
                printout(flog, 'Loading train file ' + cur_train_filename)

                cur_data, cur_labels, cur_boxes = provider.loadDataFile(cur_train_filename, point_num)
                #cur_labels = np.squeeze(cur_labels)
                #cur_labels_one_hot = convert_label_to_one_hot(cur_labels)
                print(cur_data.shape, cur_boxes.shape)

                num_data = cur_data.shape[0]
                # see http://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
                num_batch = int(-(-num_data // batch_size))

                total_loss = 0.0
                total_bbox_A_loss = 0.0
                total_bbox_B_loss = 0.0
                total_bbox_l_loss = 0.0

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = min((j + 1) * batch_size,num_data)

                    feed_dict = {
                            pointclouds_ph:     cur_data[begidx: endidx, ...], 
                            bbox_A_ph:          cur_boxes[begidx: endidx, ..., 0:2],
                            bbox_B_ph:          cur_boxes[begidx: endidx, ..., 2:4],
                            bbox_l_ph:          cur_boxes[begidx: endidx, ..., 4:5],
                            is_training_ph:     is_training 
                            }

                    _, loss_val, bbox_A_loss_val, bbox_B_loss_val, bbox_l_loss_val, bbox_A_pred_val \
                            = sess.run([train_op, loss, bbox_A_loss, bbox_B_loss, bbox_l_loss, bbox_A_pred ], feed_dict = feed_dict)

                    if (j == num_batch -1):
                        bbox_A_pred_val2 = sess.run([bbox_A_pred ], feed_dict = feed_dict)
                        print(cur_boxes[begidx: endidx, ..., 0:2], bbox_A_pred_val, bbox_A_pred_val2)
                    if epoch >= 20:
                        #import code
                        #code.interact(local=dict(globals(), **locals())) 
                        pass
        
                    #per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx, ...], axis=1)
                    #average_part_acc = np.mean(per_instance_part_acc)

                    total_loss += loss_val
                    total_bbox_A_loss += bbox_A_loss_val
                    total_bbox_B_loss += bbox_B_loss_val
                    total_bbox_l_loss += bbox_l_loss_val
                    
                   # per_instance_label_pred = np.argmax(label_pred_val, axis=1)
                   # total_label_acc += np.mean(np.float32(per_instance_label_pred == cur_labels[begidx: endidx, ...]))
                   # total_seg_acc += average_part_acc

                total_loss /= num_batch
                total_bbox_A_loss /=  num_batch
                total_bbox_B_loss /=  num_batch
                total_bbox_l_loss /=  num_batch

# total_training_loss_ph, bbox_A_training_loss_ph, bbox_B_training_loss_ph, bbox_l_training_loss_ph, training_iou_ph

                lr_sum, bn_decay_sum, batch_sum, \
                train_loss_sum, train_bbox_A_loss_sum, train_bbox_B_loss_sum, train_bbox_l_loss_sum = \
                sess.run( \
                    [lr_op, bn_decay_op, batch_op, 
                    total_train_loss_sum_op, bbox_A_training_loss_sum_op, bbox_B_training_loss_sum_op,  bbox_l_training_loss_sum_op],
                    feed_dict = {
                    total_training_loss_ph:     total_loss, 
                    bbox_A_training_loss_ph:    total_bbox_A_loss,
                    bbox_B_training_loss_ph:    total_bbox_B_loss, 
                    bbox_l_training_loss_ph:    total_bbox_l_loss
                    } )

                train_writer.add_summary(train_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_bbox_A_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_bbox_B_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_bbox_l_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(lr_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(bn_decay_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(batch_sum, i + epoch_num * num_train_file)

                printout(flog, '\tTraining Total Mean_loss: %f' % total_loss)
                printout(flog, '\t\tTraining A box loss: %f' % total_bbox_A_loss)
                printout(flog, '\t\tTraining B box loss: %f' % total_bbox_B_loss)
                printout(flog, '\t\tTraining l box loss: %f' % total_bbox_l_loss)

        def eval_one_epoch(epoch_num):
            is_training = False

            total_loss = 0.0
            total_bbox_A_loss = 0.0
            total_bbox_B_loss = 0.0
            total_bbox_l_loss = 0.0
            total_seen = 0

            #total_label_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
            #total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
            #total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)

            for i in range(num_test_file):
                cur_test_filename = os.path.join(hdf5_data_dir, test_file_list[i])
                printout(flog, 'Loading test file ' + cur_test_filename)

                cur_data, cur_labels, cur_boxes = provider.loadDataFile(cur_test_filename, point_num)
                #cur_labels = np.squeeze(cur_labels)
                #cur_labels_one_hot = convert_label_to_one_hot(cur_labels)
                print(cur_data.shape, cur_boxes.shape)

                num_data = cur_data.shape[0]
                num_batch = int(-(-num_data // batch_size))

                print("items", num_data, "batches to do", num_batch)
                total_loss = 0.0
                total_bbox_A_loss = 0.0
                total_bbox_B_loss = 0.0
                total_bbox_l_loss = 0.0
                total_seen = 0

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = min((j + 1) * batch_size, num_data)
                    feed_dict = {
                            pointclouds_ph:     cur_data[begidx: endidx, ...], 
                            bbox_A_ph:          cur_boxes[begidx: endidx, ..., 0:2],
                            bbox_B_ph:          cur_boxes[begidx: endidx, ..., 2:4],
                            bbox_l_ph:          cur_boxes[begidx: endidx, ..., 4:5],
                            is_training_ph:     is_training
                            }
                    if epoch >= 2:
                        #from pdb import set_trace; set_trace()

                        #import code
                        #code.interact(local=locals())
                        pass
                    #bbox_A_pred_val = bbox_B_pred_val = bbox_l_pred_val = np.array([[0.,0.]])
                    bbox_A_pred_val, bbox_B_pred_val, bbox_l_pred_val, loss_val, bbox_A_loss_val, bbox_B_loss_val, bbox_l_loss_val \
                    = sess.run([bbox_A_pred, bbox_B_pred, bbox_l_pred, loss, bbox_A_loss, bbox_B_loss, bbox_l_loss ], feed_dict = feed_dict)

                    # cur_test_filename
                    # cur_data[begidx: endidx, ...] -> points (BxNx3)
                    # cur_seg[begidx: endidx, ...]  -> Labels (BxN)
                    # pred_seg_res                  -> (BxN)
                    
                    total_seen += 1
                    total_loss += loss_val
                    total_bbox_A_loss += bbox_A_loss_val
                    total_bbox_B_loss += bbox_B_loss_val
                    total_bbox_l_loss += bbox_l_loss_val
                    if epoch >= 1:
                        #import code
                        #code.interact(local=locals())
                        pass

                    provider.generate_top_views_with_boxes(
                        cur_data[begidx: endidx, ...],  # BxNx3
                        cur_boxes[begidx: endidx, ...], # Bx5 (Ax,Ay,Bx,By,l)
                        np.concatenate((bbox_A_pred_val, bbox_B_pred_val, bbox_l_pred_val), axis=1),
                        cur_test_filename,
                        MODEL_STORAGE_PATH)

                    #for shape_idx in range(begidx, endidx):
                    #    total_seen_per_cat[cur_labels[shape_idx]] += 1
                    #    total_label_acc_per_cat[cur_labels[shape_idx]] += np.int32(per_instance_label_pred[shape_idx-begidx] == cur_labels[shape_idx])
                    #    total_seg_acc_per_cat[cur_labels[shape_idx]] += per_instance_part_acc[shape_idx - begidx]

            total_loss /= total_seen
            total_bbox_A_loss /=  total_seen
            total_bbox_B_loss /=  total_seen
            total_bbox_l_loss /=  total_seen

            '''
            test_loss_sum, test_bbox_A_loss_sum, test_bbox_B_loss_sum, test_bbox_l_loss_sum, test_iou = sess.run(
                [total_test_loss_sum_op, bbox_A_testing_loss_sum_op, bbox_B_testing_loss_sum_op,  bbox_l_testing_loss_sum_op, 
                            testing_iou_sum_op],
                            feed_dict = {
                            total_testing_loss_ph: total_loss, 
                            bbox_A_testing_loss_ph: total_bbox_A_loss,
                            bbox_B_testing_loss_ph: total_bbox_B_loss, 
                            bbox_l_testing_loss_ph: total_bbox_l_loss, 
                            testing_iou_ph: total_iou } )

            test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_bbox_A_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_bbox_B_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_bbox_l_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_iou, (epoch_num+1) * num_train_file-1)
            '''

            printout(flog, '\tTesting Total Mean_loss: %f' % total_loss)
            printout(flog, '\t\tTesting A box loss: %f' % total_bbox_A_loss)
            printout(flog, '\t\tTesting B box loss: %f' % total_bbox_B_loss)
            printout(flog, '\t\tTesting l box loss: %f' % total_bbox_l_loss)

            #for cat_idx in range(NUM_CATEGORIES):
            #    if total_seen_per_cat[cat_idx] > 0:
            #        printout(flog, '\n\t\tCategory %s Object Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
            #        printout(flog, '\t\tCategory %s Label Accuracy: %f' % (all_obj_cats[cat_idx][0], total_label_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))
            #        printout(flog, '\t\tCategory %s Seg Accuracy: %f' % (all_obj_cats[cat_idx][0], total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n<<< Testing on the test dataset ...')
            eval_one_epoch(epoch)

            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            train_one_epoch(train_file_idx, epoch)

            if (epoch+1) % 10 == 0:
                cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()

        flog.close()

if __name__=='__main__':
    train()
