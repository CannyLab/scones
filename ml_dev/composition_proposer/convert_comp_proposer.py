import tensorflow as tf
import state_models

def convert_state():
    TEXT_FEAT_SIZE = 300
    num_units = 64
    # model_ckpt = './checkpoints/20210318-235332/best/model.ckpt-29619'
    model_ckpt = './checkpoints/20210405-220801/model.ckpt-146217'
    s_model = state_models.SconesGPT2StateModel(num_units=num_units, coords='continuous')
    model = state_models.SconesCompositionProposerStateKerasModel(s_model.num_units, s_model.output_embed_num_units)
    
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(model_ckpt).expect_partial()

    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(session_config=session_config)

    callable = tf.function(model.greedy_decode_tfx_no_special)
    concrete_function = callable.get_concrete_function(tf.TensorSpec([None, None, 102 + TEXT_FEAT_SIZE], tf.float32, name="orig_pred"))
    tf.saved_model.save(model, '../../exported_models/composition_proposer_state/3', signatures=concrete_function)


convert_state()
