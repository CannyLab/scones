from data import word_vectors
from utils import Clipart, construct_scene_objects
import numpy as np
import state_models
import nltk
import tensorflow as tf
import os

GLOVE_MODEL = None
TEXT_FEAT_SIZE = 300
OBJ_FEAT_SIZE = 102
NUM_IDX = 58 + 2

def caption_to_vecs(data_dir, sen, pad=True, emb_size=TEXT_FEAT_SIZE):
    global GLOVE_MODEL
    if not GLOVE_MODEL:
        print("Loading Glove Model")
        GLOVE_MODEL = word_vectors.get_glove_model(data_dir)
        print("Loading Complete")
    tokens = nltk.tokenize.word_tokenize(sen.lower())
    caption = [word_vectors.get_glove(GLOVE_MODEL, token, emb_size) for token in tokens]
    vecs = np.array(caption, dtype=np.float32)
    return vecs 


def generate(model_ckpt, scenes, captions, data_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'data'), num_units=64):
    assert len(captions) == len(scenes), "Scene Array must be as long as the number of caption that modifies it"

    model = state_models.SconesGPT2StateModel(num_units=num_units)
    proposer_model = state_models.SconesCompositionProposerStateKerasModel(model.num_units, model.output_embed_num_units)

    ckpt = tf.train.Checkpoint(model=proposer_model)
    ckpt.restore(model_ckpt).expect_partial()
    
    start_token = np.zeros([1, TEXT_FEAT_SIZE + OBJ_FEAT_SIZE])
    start_token[0, 0] = 1
    end_token = np.zeros([1, TEXT_FEAT_SIZE + OBJ_FEAT_SIZE])
    end_token[0, 1] = 1

    gathered_arr = []
    for s, c in zip(scenes[-10:], captions[-10:]):
        scene_arr = np.array([art.get_array() for art in s]).reshape([-1, OBJ_FEAT_SIZE + TEXT_FEAT_SIZE])
        text_arr = np.pad(np.array(caption_to_vecs(data_dir, c)), ((0, 0), (OBJ_FEAT_SIZE, 0)))
        gathered_arr.append(start_token)
        gathered_arr.append(scene_arr)
        gathered_arr.append(end_token)
        gathered_arr.append(text_arr)
    gathered_arr.append(start_token)
    model_inputs = np.expand_dims(np.concatenate(gathered_arr, axis=0), axis=0)
    tf_outputs = proposer_model.greedy_decode_tfx_no_special(model_inputs)
    np_outputs_raw = tf_outputs.numpy()[0]
    np_outputs_arr = []
    for n in np_outputs_raw:
        if np.argmax(n[:NUM_IDX]) == 1:
            break
        np_outputs_arr.append(n) 
    new_scene = construct_scene_objects(np_outputs_arr)
    return new_scene

# Similar to the eval script, change this checkpoint path to a model checkpoint desired
model_ckpt = ''

# This following array for a scene object is arranged as follows:
# The first 60 elements is the one-hot encoding of the object's class, with the first element representing a start token, and the second element representing an end token. The rest is the actual object class from CoDraw.
# The following 35 elements is the one-hot encoding of the subclass of the object. For most objects, this is 0.
# The following 3 elements is the one-hot encoding of the size of the object.
# The following 2 elements is the one-hot encoding of the flip direction of the object.
# The following element is the normalized x-coordinate of the object.
# The following element is the normalized y-coordinate of the object
prev_cloud_arr = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.48743945360183716, 0.14592784643173218])

# After the data array is constructed, you may construct Clipart objects with it
prev_cloud_obj = Clipart(prev_cloud_arr) 

# Then, you may generate new scenes from arrays of past scenes and arrays of captions. In this example, the first scene is empty, the first users' description is 'draw a cloud in the middle'.
# The second scene has a cloud added to it, and the user's description is 'draw a sun next to the cloud. 
new_scene = generate(model_ckpt, [[], [prev_cloud_obj]], ['draw a cloud in the middle', 'draw a sun next to the cloud'])

# This new scene is returned as an array of Clipart objects