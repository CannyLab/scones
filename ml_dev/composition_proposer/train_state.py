import state_models

from datetime import datetime
import os
import tensorflow as tf 
import exporters

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_units', 64, "RNN Num Units")

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('data_dir', os.path.join(os.path.dirname(__file__), '..', '..', 'data'), 'Dataset dir')
flags.DEFINE_string('warm_start_ckpt', None, 'Ckpt to Warm-start from')
flags.DEFINE_string('model_dir', None, 'Model Dir to continue training')
flags.DEFINE_string('tfrecords_file_format', "codraw_%s_combined_state_glove.tfrecords", 'File format for tfrecords data file.')

flags.DEFINE_integer('batch_size', 64, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 600, 'Epochs to train for.')
flags.DEFINE_integer('save_every', 5, 'Save Checkpoint every x epochs')

flags.DEFINE_float('class_loss_weight', 0.05, 'Weight of Class Loss')
flags.DEFINE_float('coords_loss_weight', 1.0, 'Weight of Coords Loss')
flags.DEFINE_float('subtype_loss_weight', 0.05, 'Weight of Subtype Loss')
flags.DEFINE_float('flip_loss_weight', 0.05, 'Weight of Flip Loss')
flags.DEFINE_float('size_loss_weight', 0.05, 'Weight of Size Loss')

FLAGS = flags.FLAGS


def train():
    tf.compat.v1.disable_eager_execution()
    with tf.Graph().as_default():
        lr = FLAGS.learning_rate
        class_loss_weight = FLAGS.class_loss_weight
        subtype_loss_weight = FLAGS.subtype_loss_weight
        coords_loss_weight = FLAGS.coords_loss_weight
        size_loss_weight = FLAGS.size_loss_weight
        flip_loss_weight = FLAGS.flip_loss_weight
        epochs = FLAGS.epochs
        num_units = FLAGS.num_units

        if FLAGS.model_dir is None:
            model_dir = './checkpoints/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            model_dir = FLAGS.model_dir
        
        model_class = state_models.SconesGPT2StateModel
       
        model = model_class(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=lr),
                num_units=num_units, 
                class_loss_weight=class_loss_weight,
                subtype_loss_weight=subtype_loss_weight,
                coords_loss_weight=coords_loss_weight, 
                size_loss_weight=size_loss_weight,
                flip_loss_weight=flip_loss_weight)
        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True
        
        strategy = tf.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(session_config=session_config, train_distribute=strategy) 
        agent = tf.estimator.Estimator(model_dir=model_dir, model_fn=model.get_model_fn(), config=config, warm_start_from=FLAGS.warm_start_ckpt)
        best_exporter = exporters.BestCheckpointsExporter(compare_fn=exporters.recon_loss_smaller)

        train_steps_per_epoch = 62000 // FLAGS.batch_size
        train_spec = tf.estimator.TrainSpec(input_fn=model.get_input_fn(FLAGS.data_dir, FLAGS.batch_size,
            FLAGS.tfrecords_file_format, split="train", num_epochs=epochs), max_steps=train_steps_per_epoch * epochs)
        eval_spec = tf.estimator.EvalSpec(input_fn=model.get_input_fn(FLAGS.data_dir, FLAGS.batch_size,
            FLAGS.tfrecords_file_format, split="val", num_epochs=epochs), exporters=[best_exporter]) 

        tf.estimator.train_and_evaluate(agent, train_spec, eval_spec)


def main(_):
    train()
   
if __name__ == '__main__':
    app.run(main)

