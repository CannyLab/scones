from transformers import TFGPT2Model, GPT2Config
import tensorflow as tf
from transformers import modeling_tf_utils

import numpy as np
import utils
import sys
import os

from collections import defaultdict

EPS = 1e-6 

NUM_CLASSES = 58
NUM_IDX = NUM_CLASSES + 2
NUM_SUBTYPE = 35
NUM_SIZE = 3
NUM_FLIP = 2

MAX_NUM_TOKENS = 128
NUM_CATS = NUM_IDX + NUM_SUBTYPE + NUM_SIZE + NUM_FLIP
NUM_NUM = 2

DEBUG = False

class SconesCompositionProposerStateKerasModel(tf.keras.Model):
    def __init__(self, num_units, output_embed_num_units):
        super().__init__()
        self.config = GPT2Config(vocab_size=1, n_positions=1024, n_ctx=1024, n_embd=num_units, n_layer=6, n_head=8)
        self.input_embedding = tf.keras.layers.Dense(num_units)  
        self.transformer = TFGPT2Model(self.config)
        self.output_embedding = tf.keras.layers.Dense(output_embed_num_units)
        self.text_idx_embedding = tf.keras.layers.Embedding(MAX_NUM_TOKENS,
            self.config.n_embd,
            embeddings_initializer=modeling_tf_utils.get_initializer(self.config.initializer_range))

        self.obj_idx_embedding = tf.keras.layers.Embedding(MAX_NUM_TOKENS,
                self.config.n_embd,
                embeddings_initializer=modeling_tf_utils.get_initializer(self.config.initializer_range))

    def call(self, pred, decoder_mask=None, turn_idxs=None, token_types=None, token_idxs=None, training=True, past=None, new_pos_embeddings=False):
        pred = self.input_embedding(pred)
        if new_pos_embeddings:
            pred += self.text_idx_embedding(token_types * token_idxs) + self.obj_idx_embedding((1 - token_types) * token_idxs)
            transformer_output, past = self.transformer(inputs=None, inputs_embeds=pred, attention_mask=decoder_mask, past=past, position_ids=turn_idxs, training=training)
        else:
            transformer_output, past = self.transformer(inputs=None, inputs_embeds=pred, attention_mask=decoder_mask, past=past, training=training)
        output = self.output_embedding(transformer_output)
        return output, past 

    def greedy_decode(self, orig_pred, decoder_mask, turn_idxs, token_types, token_idxs, past=None, coords="continuous", max_length=16, new_pos_embeddings=False):
        cur_len = 0
        pred_start_shape = tf.shape(orig_pred)[1]
        pred = orig_pred
        while cur_len < max_length:
            outputs, past = self(pred, decoder_mask, turn_idxs, token_types, token_idxs, training=False, new_pos_embeddings=new_pos_embeddings)#, past)
            next_token_vals = utils.reprocess_outputs(outputs, 1, emb_size=300)[:, -1:, :]
            pred = tf.concat([pred, next_token_vals], 1)

            turn_idxs = tf.concat([turn_idxs, turn_idxs[:, -1:]], axis=1)
            token_idxs = tf.concat([token_idxs, token_idxs[:, -1:] + 1], axis=1)
            token_types = tf.concat([token_types,  token_types[:, -1:]], axis=1)
            decoder_mask = tf.concat([decoder_mask, tf.ones_like(decoder_mask[:, :1])], axis=1)
            cur_len = cur_len + 1


        return pred[:, pred_start_shape:]

    def greedy_decode_tfx(self, orig_pred, turn_idxs, token_types, token_idxs):
        return self.greedy_decode(orig_pred, tf.ones_like(turn_idxs), turn_idxs, token_types, token_idxs, new_pos_embeddings=True)

    def greedy_decode_tfx_no_special(self, orig_pred):
        return self.greedy_decode(orig_pred, tf.ones(tf.shape(orig_pred)[:2]), tf.zeros([1, 1]), tf.zeros([1, 1]), tf.zeros([1, 1]), new_pos_embeddings=False)

     
class SconesGPT2StateModel():
    def __init__(self,
                 optimizer=None,
                 num_units=64,
                 num_layers=6,
                 num_classes=58,
                 class_loss_weight=1.0,
                 coords_loss_weight=1.0,
                 subtype_loss_weight=1.0,
                 size_loss_weight=1.0,
                 flip_loss_weight=1.0):
        
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.num_units = num_units
        self.num_grids = 1

        self.coords_loss_weight = coords_loss_weight
        self.class_loss_weight = class_loss_weight
        self.subtype_loss_weight = subtype_loss_weight
        self.size_loss_weight = size_loss_weight
        self.flip_loss_weight = flip_loss_weight
        
        self.output_embed_num_units = NUM_CATS + self.num_grids * NUM_NUM
        self.keras_model_cls = SconesCompositionProposerStateKerasModel

    def process_inputs(self, inputs, mode):
        pred = inputs['combined_vecs']    
        decoder_mask = tf.sequence_mask(inputs['combined_len'] - 1, tf.shape(pred)[-2], dtype=tf.float32)
        turn_idxs, token_types, token_idxs = inputs['turn_idxs'], inputs['token_types'], inputs['token_idxs']
        outputs = [pred, decoder_mask, turn_idxs, token_types, token_idxs]
        if mode == tf.estimator.ModeKeys.PREDICT:
            outputs = [tf.expand_dims(o, 0) for o in outputs]
        return outputs


    def get_input_fn(self, data_dir, batch_size, train_file_name, split="train", shuffle=True, num_epochs=1):
        def _input_fn():  
            is_training = split == 'train'
            fnames = [os.path.join(data_dir, train_file_name % split)]
            dataset = tf.data.TFRecordDataset(fnames)
            parse_fn = lambda x: utils.parse_data_state(x)
            dataset = dataset.map(parse_fn)

            if shuffle:
                dataset = dataset.shuffle(buffer_size=5000)
            
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)
            return dataset
        return _input_fn

    def get_model_fn(self):
        def _model_fn(features, labels, mode, params):       
            proposer_model = self.keras_model_cls(self.num_units, self.output_embed_num_units)
            global_step = tf.compat.v1.train.get_or_create_global_step()  
            inputs = features
            model_inputs = self.process_inputs(inputs, mode)
            
            output, _ = proposer_model(*model_inputs, training=mode != tf.estimator.ModeKeys.EVAL, new_pos_embeddings=False)
            
            if mode == tf.estimator.ModeKeys.PREDICT:
                return NotImplementedError("Prediction is done directly on the nested Keras model")

            loss, output_tensors = self.loss_function(inputs, output)
            
            if mode == tf.estimator.ModeKeys.EVAL:
                train_op = None
                ckpt = tf.train.Checkpoint(model=proposer_model, global_step=global_step)
                scaffold = tf.compat.v1.train.Scaffold(saver=ckpt)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold, eval_metric_ops={'Total_Loss': tf.compat.v1.metrics.mean(loss)})
            else:
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
                gvs = self.optimizer.compute_gradients(loss, var_list=var_list)
                train_op = self.optimizer.apply_gradients(gvs, global_step=global_step)

                ckpt = tf.train.Checkpoint(model=proposer_model, optimizer=self.optimizer, global_step=global_step)
                scaffold = tf.compat.v1.train.Scaffold(saver=ckpt)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)

        return _model_fn

    def get_class_loss(self, gen_classes, gt_classes, class_mask):
        class_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(gt_classes, gen_classes) * class_mask, axis=-1) / (tf.reduce_sum(class_mask, axis=-1) + EPS) 
        return class_loss

    def get_coords_loss(self, new_coords_x, new_coords_y, gt_coords, coords_mask):
        total_coords = tf.reduce_sum(coords_mask, axis=-1) + EPS

        coords_x = tf.math.sigmoid(new_coords_x) * 2 - 0.5
        coords_y = tf.math.sigmoid(new_coords_y) * 2 - 0.5
        coords = tf.concat([coords_x, coords_y], axis=-1)
        clipped_coords = tf.clip_by_value(coords, 0.0, 1.0)
        display_coords = tf.cast(clipped_coords * 19.9, tf.int32)
        pre_mask_loss = tf.reduce_sum((coords - gt_coords) ** 2, axis=-1)
        masked_loss = pre_mask_loss * coords_mask
        coords_loss = tf.reduce_sum(masked_loss, axis=-1) / total_coords
        return coords_loss, display_coords

    def loss_function(self, inputs, outputs):
        max_seq_len = tf.shape(outputs)[1] - 1
        with_last_mask = tf.sequence_mask(inputs['combined_len'] - 1, maxlen=max_seq_len, dtype=tf.float32)
        no_last_mask = tf.sequence_mask(inputs['combined_len'] - 2, maxlen=max_seq_len, dtype=tf.float32)
        input_mask = tf.sequence_mask(inputs['combined_input_len'], maxlen=max_seq_len, dtype=tf.float32)

        class_mask = with_last_mask - input_mask
        obj_mask = no_last_mask - input_mask
        with tf.name_scope('loss'):
            new_coords_x, new_coords_y, new_sizes, new_flips, new_classes, new_subtypes = [params for params in utils.split_state(outputs[:, :-1], self.num_grids, num_classes=self.num_classes)]
          
            gt_classes = inputs['combined_vecs'][:, 1:, :NUM_IDX]
            gt_subtypes = inputs['combined_vecs'][:, 1:, NUM_IDX:NUM_IDX + NUM_SUBTYPE]
            
            gt_sizes = inputs['combined_vecs'][:, 1:, NUM_IDX + NUM_SUBTYPE:NUM_IDX + NUM_SUBTYPE + NUM_SIZE]
            gt_flips = inputs['combined_vecs'][:, 1:, NUM_IDX + NUM_SUBTYPE + NUM_SIZE:NUM_IDX + NUM_SUBTYPE + NUM_SIZE + NUM_FLIP]
            gt_coords = inputs['combined_vecs'][:, 1:, NUM_CATS:NUM_CATS + NUM_NUM]
        
            class_loss = self.get_class_loss(new_classes, gt_classes, class_mask)
            subtype_loss = self.get_class_loss(new_subtypes, gt_subtypes, obj_mask)
            size_loss = self.get_class_loss(new_sizes, gt_sizes, obj_mask)
            flip_loss = self.get_class_loss(new_flips, gt_flips, obj_mask)
            coords_loss, _ = self.get_coords_loss(new_coords_x, new_coords_y, gt_coords, obj_mask)
            total_loss = self.coords_loss_weight * tf.reduce_mean(coords_loss) + self.class_loss_weight * tf.reduce_mean(class_loss) + self.size_loss_weight * tf.reduce_mean(size_loss) + self.flip_loss_weight * tf.reduce_mean(flip_loss) + self.subtype_loss_weight * tf.reduce_mean(subtype_loss)
            
            loss_dict = {'Total_Loss': total_loss,
                         'Coords_Loss': tf.reduce_mean(coords_loss),
                         'Size_Loss': tf.reduce_mean(size_loss),
                         'Flip_Loss': tf.reduce_mean(flip_loss),
                         'Class_Loss': tf.reduce_mean(class_loss),
                         'Subtype_Loss': tf.reduce_mean(subtype_loss),
                         }

            return total_loss, loss_dict
           
