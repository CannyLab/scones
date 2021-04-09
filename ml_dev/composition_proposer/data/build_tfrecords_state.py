import nltk
import os, sys

from word_vectors import get_glove_model, get_glove
import numpy as np
import tensorflow as tf
from collections import defaultdict

from tqdm import tqdm
import json
import random

WORD_EMBED_SIZE = 300

MAX_LEN_INPUT = 440
MAX_LEN_FINAL = 17

WIDTH = 500.
HEIGHT = 400.

STEP_SIZE = 10
NUM_IDX = 58 + 2
NUM_SUBTYPE = 35
NUM_EXPRESSION = 5
NUM_DEPTH = 3
NUM_FLIP = 2

NUM_ACTIONS = 4
NUM_SELECTIONS = 17

NUM_CATS = NUM_IDX + NUM_SUBTYPE + NUM_DEPTH + NUM_FLIP
NUM_NUM = 2 # x, y

GLOVE_MODEL = None


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64s_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(emb):
    return tf.train.Feature(float_list=tf.train.FloatList(value=emb))

def get_max_len(examples):
    max_len = 0
    for i in examples:
        c = i[3]
        tokens = nltk.tokenize.word_tokenize(c.lower())
        l = len(tokens) 
        if l > max_len:
            max_len = l
    return max_len


def caption_to_vecs(data_dir, sen, max_len, pad=True, emb_size=WORD_EMBED_SIZE):
    global GLOVE_MODEL
    if not GLOVE_MODEL:
        GLOVE_MODEL = get_glove_model(data_dir)
    tokens = nltk.tokenize.word_tokenize(sen.lower())
    caption = [get_glove(GLOVE_MODEL, token, emb_size) for token in tokens]
    caption_len = len(caption)
    if pad:
        caption.extend([np.zeros((emb_size,)) for i in range(caption_len, max_len)])

    vecs = np.array(caption, dtype=np.float32)
    return vecs, caption_len # Pad Zeros for ids representation


def get_clipart_idx(clipart_obj_idx, clipart_type_idx):
    total_pos = [0, 8, 18, 19, 20, 26, 36, 43]

    offset = int(clipart_obj_idx)
    if (int(clipart_type_idx) == 2) or (int(clipart_type_idx) == 3):
        offset = 0
    return total_pos[int(clipart_type_idx)] + offset


def load_obj_arr(cur_obj):
    box = np.zeros((WORD_EMBED_SIZE + NUM_NUM + NUM_CATS,), dtype=np.float32)
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
    return box


def load_examples(data_dir):
    with open(os.path.join(data_dir, 'CoDraw_1_0.json'), 'r') as f:
        n = json.load(f)
        vals = n['data'].items()

    examples = []
    vals = list(vals)
    random.shuffle(vals)
    train_vals = [(k, v) for k, v in vals if 'train' in k]
    val_vals = [(k, v) for k, v in vals if 'val' in k]
    test_vals = [(k, v) for k, v in vals if 'test' in k]
    for values in [train_vals, val_vals, test_vals]:
        cur_examples = []
        for (key, i) in tqdm(values):
            ds = i['dialog']
            abs_t = i['abs_t']
            for d in ds:
                abs_b = d['abs_b']
                abs_d = d['abs_d']
                caption = d['msg_t']
                orig_arr = abs_b.split(',')[1:-1]
                new_arr = abs_d.split(',')[1:-1]
                final_arr = abs_t.split(',')[1:]
                orig_vecs_arr = []
                new_vecs_arr = []
                final_vecs_arr = []

                for o_arr, v_arr in [(orig_arr, orig_vecs_arr), (new_arr, new_vecs_arr), (final_arr, final_vecs_arr)]:
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
                
                exp = (int(d['seq_t']), int(d['seq_d']), int(key.split('_')[-1]), caption, orig_vecs_arr, new_vecs_arr)
                cur_examples.append(exp)
        examples.append(cur_examples)
    return examples


def write_tfrecords(data_dir, examples, tfrecords_format='codraw_{}_combined_state_glove.tfrecords', split='train', write=True):
    global NUM_NUM, NUM_CATS, NUM_IDX
    
    tfrecords_filename = os.path.join(data_dir, tfrecords_format.format(split))
    if write:
        writer = tf.io.TFRecordWriter(tfrecords_filename)

    max_len = get_max_len(examples)
    agg = []
    prev_example_id = -1
    embeds = [caption_to_vecs(data_dir, e[3], max_len, pad=False) for e in examples]
    for idx, e in tqdm(enumerate(examples)):
        seq_t = e[0]
        seq_d = e[1]
        example_id = e[2]
        caption = e[3]
        if example_id != prev_example_id:
            agg = []
            prev_example_id = example_id
        input_object_features = e[4]
        final_object_features = e[5]
        
        box_start_token = np.zeros((WORD_EMBED_SIZE + NUM_NUM + NUM_CATS), dtype=np.float32)
        box_start_token[0] = 1
        box_end_token = np.zeros((WORD_EMBED_SIZE + NUM_NUM + NUM_CATS), dtype=np.float32)
        box_end_token[1] = 1

        input_values = []
        final_values = []
        
        for feats, vals in [(input_object_features, input_values), (final_object_features, final_values)]:
            vals.append(box_start_token)
            for cur_obj in feats:
                box = load_obj_arr(cur_obj)
                vals.append(box)
            vals.append(box_end_token)

        c_vecs, _ = embeds[idx] 

        text_values = []
        for c in c_vecs:
            text = np.zeros((WORD_EMBED_SIZE + NUM_NUM + NUM_CATS,), dtype=np.float32)
            text[-WORD_EMBED_SIZE:] = c
            text_values.append(text)
        
        total_input = MAX_LEN_INPUT 
        current_scene_len = len(input_values)

        token_idxs = [i + 1 for i in range(current_scene_len)]
        token_types = [0] * len(token_idxs)

        input_values.extend(text_values)
        token_idxs.extend([i + 1 for i in range(len(text_values))])
        token_types.extend([1] * len(text_values))
        agg.append((input_values, token_types, token_idxs))
        
        input_values = []
        token_types = []
        token_idxs = []
        turn_idxs = []

        for si, step in enumerate(agg[-STEP_SIZE:]):
            turn_idx = min(STEP_SIZE, len(agg)) - si + 1 
            input_values.extend(step[0])
            token_types.extend(step[1])
            token_idxs.extend(step[2])
            turn_idxs.extend([turn_idx] * len(step[2]))
        
        combined_input_len = len(input_values)
        input_values.extend(final_values)
        combined_len = len(input_values)

        token_types.extend([0] * len(final_values))
        token_idxs.extend([i + 1 for i in range(len(final_values))])
        turn_idxs.extend([0] * len(final_values))
        
        assert len(token_idxs) == len(token_types) 
        assert len(turn_idxs) == len(token_types)
        assert len(token_types) == combined_len

        input_values = np.array(input_values)
        token_types = np.array(token_types, dtype=np.int32)
        token_idxs = np.array(token_idxs, dtype=np.int32)
        turn_idxs = np.array(turn_idxs, dtype=np.int32)

        pad_amount = total_input - combined_len
        input_values = np.pad(input_values, [[0, pad_amount], [0, 0]], 'constant')
        token_types = np.pad(token_types, [[0, pad_amount]], 'constant')
        token_idxs = np.pad(token_idxs, [[0, pad_amount]], 'constant')
        turn_idxs = np.pad(turn_idxs, [[0, pad_amount]], 'constant')


        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_t': _int64_feature(seq_t),
            'seq_d': _int64_feature(seq_d),
            'example_id': _int64_feature(example_id),
            'combined_vecs': _bytes_feature(input_values.tostring()),
            'current_scene_len': _int64_feature(current_scene_len),
            'combined_input_len': _int64_feature(combined_input_len),
            'combined_len': _int64_feature(combined_len),
            'turn_idxs': _bytes_feature(turn_idxs.tostring()),
            'token_types': _bytes_feature(token_types.tostring()),
            'token_idxs': _bytes_feature(token_idxs.tostring())
        }))

        if write:
            writer.write(example.SerializeToString())

    if write:
        writer.close()


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
    train_examples, val_examples, test_examples = load_examples(data_dir)
    write_tfrecords(data_dir, train_examples, tfrecords_format='codraw_{}_combined_state_glove.tfrecords')
    write_tfrecords(data_dir, val_examples, tfrecords_format='codraw_{}_combined_state_glove.tfrecords', split='val')
    write_tfrecords(data_dir, test_examples, tfrecords_format='codraw_{}_combined_state_glove.tfrecords', split='test')
  
