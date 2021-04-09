from eval import eval_state_data, eval_state_codraw
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
s, g = eval_state_data(data_dir, num_examples=512, split='test', tfrecords_file_format="codraw_%s_combined_state_glove.tfrecords")

model_ckpt = './checkpoints/20210405-220801/model.ckpt-146217'
score_data, scene_data = eval_state_codraw(model_ckpt, s, g)