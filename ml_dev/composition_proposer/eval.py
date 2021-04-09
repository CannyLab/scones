import state_models
from datetime import datetime
import json
import os
import tensorflow as tf 

import numpy as np
from utils import load_gt, construct_scene_objects

TEXT_FEAT_SIZE = 300
NUM_IDX = 58 + 2

def scene_similarity(pred, target):
    idx1 = set(x.idx for x in target)
    idx2 = set(x.idx for x in pred)

    intersection_size = len(idx1 & idx2)
    union_size = len(idx1 | idx2)

    common_idxs = list(idx1 & idx2)
    match1 = [[x for x in target if x.idx == idx][0] for idx in common_idxs]
    match2 = [[x for x in pred if x.idx == idx][0] for idx in common_idxs]

    num = np.zeros(8)
    denom = np.zeros(8)

    num[0] = intersection_size

    for c1, c2 in zip(match1, match2):
        num[1] += int(c1.flip != c2.flip)
        if c1.idx in [18, 19]:
            num[2] += int(c1.expression != c2.expression)
            num[3] += int(c1.pose != c2.pose)
        num[4] += int(c1.depth != c2.depth)
        num[5] += min(1.0, np.sqrt((c1.normed_x - c2.normed_x) ** 2 + (c1.normed_y - c2.normed_y) ** 2))

    denom[:6] = union_size

    for idx_i in range(len(match1)):
        for idx_j in range(idx_i, len(match1)):
            if idx_i == idx_j:
                continue
            c1i, c1j = match1[idx_i], match1[idx_j]
            c2i, c2j = match2[idx_i], match2[idx_j]

            # TODO(nikita): this doesn't correctly handle the case if two
            # cliparts have *exactly* the same x/y coordinates in the target
            num[6] += int((c1i.x - c1j.x) * (c2i.x - c2j.x) <= 0)
            num[7] += int((c1i.y - c1j.y) * (c2i.y - c2j.y) <= 0)

    denom[6:] = union_size * (intersection_size - 1)

    denom = np.maximum(denom, 1)

    score_components = num / denom
    score_weights = np.array([5,-1,-0.5,-0.5,-1,-1,-1,-1])
    return score_components @ score_weights

def eval_state_data(data_dir, num_examples=128, batch_size=32, tfrecords_file_format="codraw_%s_combined_state_glove.tfrecords", split='train'):
    model = state_models.SconesGPT2StateModel(num_units=64)
    input_dataset = model.get_input_fn(os.path.join(data_dir), batch_size,
            tfrecords_file_format, split=split, shuffle=False)()
    dataset_iter = iter(input_dataset)

    with open(os.path.join(data_dir, 'CoDraw_1_0.json')) as f:
        data_json = json.load(f)

    cur_id = -1
    seqs = []
    gts = []
    cur_seq = []
    read_examples = 0
    for _ in range(num_examples):
        try:
            examples = next(dataset_iter)[0]
        except StopIteration:
            break
        data = examples['example_id'], examples['combined_len'], examples['current_scene_len'], examples['combined_vecs'], examples['seq_d'], examples['seq_t']
        for i in range(len(data[0])):
            if cur_id != int(data[0][i]):
                prev_id = cur_id
                cur_id = int(data[0][i])
                if cur_seq:
                    gts.append(load_gt(prev_id, data_json, split))
                    seqs.append(cur_seq)
                    cur_seq = []
            cur_step = {k: v[i].numpy() for k, v in examples.items()}
            cur_seq.append(cur_step)
        
    prev_id = cur_id
    gts.append(load_gt(prev_id, data_json, split))
    seqs.append(cur_seq) 
    return seqs, gts
    

def eval_state_codraw(model_ckpt, seqs, gts=None, num_units=64, num_classes=58, steps=10):
    def extract_current_caption(combined_vecs, token_types, turn_idxs):
        cur_max_ti = 10
        cap_arr = []
        for i, ti in enumerate(turn_idxs):
            if token_types[i] == 1:
                if ti > 0 and ti < cur_max_ti:
                    cur_max_ti = ti
                    cap_arr = []
                if ti == cur_max_ti:
                    cap_arr.append(i)
        return combined_vecs[min(cap_arr):max(cap_arr) + 1]


    model = state_models.SconesGPT2StateModel(num_units=num_units)
    proposer_model = state_models.SconesCompositionProposerStateKerasModel(model.num_units, model.output_embed_num_units)
    
    ckpt = tf.train.Checkpoint(model=proposer_model)
    ckpt.restore(model_ckpt).expect_partial()

    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    real_scores = []
    low_score_data = []
    for idx, s in enumerate(seqs):
        print(idx)
        outputs = []
        step_data = []
        input_agg = {
            'combined_vecs': [],
            'turn_idxs': [],
            'token_types': [],
            'token_idxs': []
        }
        first = True
        start_token = None
        end_token = None

        for step in s:
            outputs = np.array(outputs, dtype=np.float32)
            existing_len = step['combined_input_len']
            if first:
                first = False
                step_captions = None
                input_agg['combined_vecs'].append(step['combined_vecs'][:existing_len])
                start_token = step['combined_vecs'][:1]
                end_token = step['combined_vecs'][1:2]
                input_agg['turn_idxs'].append(step['turn_idxs'][:existing_len])
                input_agg['token_types'].append(step['token_types'][:existing_len])
                input_agg['token_idxs'].append(step['token_idxs'][:existing_len])
            else:
                outputs = outputs.reshape((-1, 102 + TEXT_FEAT_SIZE))
                step_captions = extract_current_caption(step['combined_vecs'], step['token_types'], step['turn_idxs'])
                step_combined_vecs = np.concatenate([start_token, outputs, end_token, step_captions], axis=0)
                input_agg['turn_idxs'] = [x + 1 for x in input_agg['turn_idxs']]
                input_agg['combined_vecs'].append(step_combined_vecs)
                input_agg['turn_idxs'].append(np.array([2] * step_combined_vecs.shape[0]))
                input_agg['token_types'].append(np.concatenate([np.array([0] * (outputs.shape[0] + 2)), np.array([1] * step_captions.shape[0])], axis=0))
                input_agg['token_idxs'].append(np.concatenate([np.array(list(range(outputs.shape[0] + 2))) + 1, np.array(list(range(step_captions.shape[0])))], axis=0))
                
            new_step = {
                'combined_vecs': np.concatenate(input_agg['combined_vecs'][-steps:] + [start_token], axis=0),
                'turn_idxs': np.concatenate(input_agg['turn_idxs'][-steps:] + [np.array([0])], axis=0),
                'token_types': np.concatenate(input_agg['token_types'][-steps:] + [np.array([0])], axis=0),
                'token_idxs': np.concatenate(input_agg['token_idxs'][-steps:] + [np.array([1])], axis=0),
            }
            new_step['combined_len'] = new_step['combined_vecs'].shape[0] + 32
           
            model_inputs = model.process_inputs(new_step, tf.estimator.ModeKeys.PREDICT)
            output = proposer_model.greedy_decode(*model_inputs, new_pos_embeddings=False)
            nn_output = output.numpy()[0]
            outputs = []
        
            for n in nn_output:
                if np.argmax(n[:NUM_IDX]) == 1:
                    break
                outputs.append(n) 
            new_scene = construct_scene_objects(outputs)

            gt_scene = construct_scene_objects(step['combined_vecs'][step['combined_input_len'] + 1:step['combined_len'] - 1])
            input_scene = step_captions
            sd = [[step['seq_d'], step['seq_t'], step['example_id']], input_scene, gt_scene, new_scene, -1]
            step_data.append(sd)
            
        if gts is not None:
            real_gt, _ = gts[idx]
            
        cur_score = scene_similarity(construct_scene_objects(outputs), construct_scene_objects(real_gt))
        real_scores.append(cur_score)
        
        print('Cur Score: ', real_scores[-1])
        print('Score: ', sum(real_scores) / len(real_scores))

    return real_scores, low_score_data
