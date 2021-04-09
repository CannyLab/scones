import os, sys
import math
import numpy as np
import tensorflow as tf
import random

WIDTH = 500.
HEIGHT = 400.

TEXT_FEAT_SIZE = 300

NUM_IDX = 58 + 2
NUM_SUBTYPE = 35
NUM_DEPTH = 3
NUM_FLIP = 2

NUM_CATS = NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP
NUM_NUM = 2
NUM_ACTIONS = 4
NUM_SELECTIONS = 17

NUM_EXPRESSION = 5

NUM_GRIDS = NUM_GRIDS_X = NUM_GRIDS_Y = 20


def split_state(inputs, num_grids, num_classes=58):
    NUM_IDX = num_classes + 2
    NUM_CATS = NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP
    classes = inputs[:, :, :NUM_IDX] #36] 126 classes + start, end
    subtype = inputs[:, :, NUM_IDX:NUM_IDX+NUM_SUBTYPE]
    size = inputs[:, :, NUM_IDX+NUM_SUBTYPE:NUM_IDX + NUM_SUBTYPE + NUM_DEPTH]
    flip = inputs[:, :, NUM_IDX + NUM_SUBTYPE + NUM_DEPTH: NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP]
    coords_x = inputs[:, :, NUM_CATS:NUM_CATS + num_grids]
    coords_y = inputs[:, :, NUM_CATS + num_grids:NUM_CATS + num_grids * 2]
    return coords_x, coords_y, size, flip, classes, subtype


def reprocess_outputs(*args, emb_size=100, **kwargs):
    coords_x, coords_y, size, flip, classes, subtype = split_state(*args, **kwargs)
    coords_x = tf.math.sigmoid(coords_x) * 2 - 0.5
    coords_y = tf.math.sigmoid(coords_y) * 2 - 0.5
    new_tensors_arr = [tf.one_hot(tf.math.argmax(classes, axis=-1), NUM_IDX), tf.one_hot(tf.math.argmax(subtype, axis=-1), NUM_SUBTYPE), 
        tf.one_hot(tf.math.argmax(size, axis=-1), NUM_DEPTH), tf.one_hot(tf.math.argmax(flip, axis=-1), NUM_FLIP), coords_x, coords_y]
    return tf.pad(tf.concat(new_tensors_arr, axis=-1), [[0, 0], [0, 0], [0, emb_size]])


def parse_data_state(serialized_example, is_training=True, emb_size=300, deploy=True):
    
    features = tf.io.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={          
        'seq_t': tf.io.FixedLenFeature([], tf.int64),
        'seq_d': tf.io.FixedLenFeature([], tf.int64),
        'example_id': tf.io.FixedLenFeature([], tf.int64),
        'combined_vecs': tf.io.FixedLenFeature([], tf.string),
        'current_scene_len': tf.io.FixedLenFeature([], tf.int64),
        'combined_input_len': tf.io.FixedLenFeature([], tf.int64),
        'combined_len': tf.io.FixedLenFeature([], tf.int64),
        'turn_idxs': tf.io.FixedLenFeature([], tf.string),
        'token_types': tf.io.FixedLenFeature([], tf.string),
        'token_idxs': tf.io.FixedLenFeature([], tf.string)
        })

    combined_objects = tf.io.decode_raw(features['combined_vecs'], tf.float32)
    combined_objects = tf.reshape(combined_objects, [-1, NUM_CATS + NUM_NUM + TEXT_FEAT_SIZE]) # 336
    
    turn_idxs = tf.io.decode_raw(features['turn_idxs'], tf.int32)
    turn_idxs = tf.reshape(turn_idxs, [-1])

    token_types = tf.io.decode_raw(features['token_types'], tf.int32)
    token_types = tf.reshape(token_types, [-1])

    token_idxs = tf.io.decode_raw(features['token_idxs'], tf.int32)
    token_idxs = tf.reshape(token_idxs, [-1])

    data = {
        'seq_t': features['seq_t'],
        'seq_d': features['seq_d'],
        'example_id': features['example_id'],
        'combined_vecs': combined_objects,
        'current_scene_len': features['current_scene_len'],
        'combined_input_len': features['combined_input_len'],
        'combined_len': features['combined_len'],
        'turn_idxs': turn_idxs,
        'token_types': token_types,
        'token_idxs': token_idxs
    }
            
    return (data, tf.constant([0., 0.]))


def get_clipart_idx(clipart_obj_idx, clipart_type_idx):
    total_pos = [0, 8, 18, 19, 20, 26, 36, 43]

    offset = int(clipart_obj_idx)
    if (int(clipart_type_idx) == 2) or (int(clipart_type_idx) == 3):
        offset = 0
    return total_pos[int(clipart_type_idx)] + offset


class Clipart(object):
    def __init__(self, arr, num_classes=58, mode='orig'):
        NUM_IDX = num_classes + 2   
        NUM_CATS = NUM_IDX 
        self.real_start_end_idx = np.argmax(arr[:NUM_IDX], axis=-1)
        self.idx = np.argmax(arr[2:NUM_IDX], axis=-1)
        if mode == 'orig':
            NUM_CATS += NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP
            self.subtype = np.argmax(arr[NUM_IDX:NUM_IDX+NUM_SUBTYPE], axis=-1)
            self.depth = np.argmax(arr[NUM_IDX+NUM_SUBTYPE:NUM_IDX+NUM_SUBTYPE+NUM_DEPTH], axis=-1)
            self.flip = np.argmax(arr[NUM_IDX+NUM_SUBTYPE+NUM_DEPTH:NUM_IDX+NUM_SUBTYPE+NUM_DEPTH+NUM_FLIP], axis=-1)
        else:
            NUM_CATS = NUM_IDX + NUM_DEPTH 
            self.subtype = 0
            self.flip = 0
            self.depth = np.argmax(arr[NUM_IDX:NUM_IDX + NUM_DEPTH])
        
        self.normed_x = arr[NUM_CATS] 
        self.normed_y = arr[NUM_CATS+1] 
        self.x = self.normed_x * WIDTH
        self.y = self.normed_y * HEIGHT
    
        self.pose = self.subtype // NUM_EXPRESSION
        self.expression = self.subtype % NUM_EXPRESSION
        
    def get_array(self, num_classes=58):
        NUM_IDX = num_classes + 2   
        NUM_CATS = NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP
        NUM_NUM = 2
        
        output_arr = np.zeros((NUM_CATS + NUM_NUM + TEXT_FEAT_SIZE,))
        output_arr[self.real_start_end_idx] = 1
        output_arr[self.subtype + NUM_IDX] = 1
        output_arr[self.depth + NUM_IDX + NUM_SUBTYPE] = 1
        output_arr[self.flip + NUM_IDX + NUM_SUBTYPE + NUM_DEPTH] = 1
        output_arr[NUM_CATS] = self.normed_x
        output_arr[NUM_CATS + 1] = self.normed_y
        return output_arr


def load_single_example(o_arr):
    v_arr = []
    for i in range(0, len(o_arr), 8):
        cur_obj = o_arr[i:i+8]
        if float(cur_obj[4]) > -10000 and float(cur_obj[5]) > -10000:
            cur_obj_arr = [0, 0, 0, 0, 0, 0, 0]
            cur_obj_arr[0] = get_clipart_idx(cur_obj[2], cur_obj[3])
            cur_obj_arr[1] = cur_obj[3]
            cur_obj_arr[2] = 0
            if (int(cur_obj[3]) == 2) or (int(cur_obj[3]) == 3):
                cur_obj_arr[2] = cur_obj[2]

            cur_obj_arr[3] = cur_obj[6]
            cur_obj_arr[4] = cur_obj[7]

            cur_obj_arr[5] = float(round(float(cur_obj[4]))) / WIDTH                     
            cur_obj_arr[6] = float(round(float(cur_obj[5]))) / HEIGHT

            v_arr.append(cur_obj_arr)
    
    return v_arr


def convert_example_tfrecords(v_arr):
    vals = []
    for cur_obj in v_arr:
        box = np.zeros((TEXT_FEAT_SIZE + NUM_NUM + NUM_CATS,), dtype=np.float32)
        art_idx = int(cur_obj[0])
        subtype_idx = int(cur_obj[2])
        depth = int(cur_obj[3])
        flip = int(cur_obj[4])
        x = cur_obj[5]
        y = cur_obj[6]
        box[art_idx + 2] = 1
        box[NUM_IDX + subtype_idx] = 1
        box[NUM_IDX + NUM_SUBTYPE + depth] = 1
        box[NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + flip] = 1
        box[NUM_CATS] = x
        box[NUM_CATS + 1] = y
        vals.append(box)
    return vals


def load_gt(example_id, data_json, split):
    abs_t = data_json['data'][split + '_' + str(example_id).zfill(5)]['abs_t']
    data_arr = abs_t.split(',')[1:]
    example_arr = load_single_example(data_arr)
    gtt = convert_example_tfrecords(example_arr)

    abs_d = data_json['data'][split + '_' + str(example_id).zfill(5)]['dialog'][-1]['abs_d']
    data_arr = abs_d.split(',')[1:-1]
    example_arr = load_single_example(data_arr)
    gtd = convert_example_tfrecords(example_arr)

    return gtt, gtd
        
        
def construct_scene_objects(objs, num_classes=58, mode='orig'):
    return [Clipart(o, num_classes, mode) for o in objs]


def unprocess_possible_actions(arr, num_classes=58, mode='orig'):
    NUM_IDX = num_classes + 2   
    if mode == 'orig':
        NUM_CATS = NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP
    elif mode == 'simple':
        NUM_CATS = NUM_IDX + NUM_DEPTH

    possible_actions = []
    for i in range(arr.shape[0]):
        action_arr = arr[i, NUM_CATS + NUM_NUM: NUM_CATS+NUM_NUM+NUM_ACTIONS]
        selection_arr = arr[i, NUM_CATS+NUM_NUM+NUM_ACTIONS:]
        not_action = sum(action_arr) == 0
        if not_action:
            continue
        
        selection = np.argmax(selection_arr)
        action = np.argmax(action_arr)
        clipart = Clipart(arr[i], mode=mode)

        possible_actions.append({
            'action': action,
            'clipart': clipart,
            'selection': selection
        })
    return possible_actions
   