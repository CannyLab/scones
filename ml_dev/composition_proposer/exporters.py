import tensorflow as tf
from collections import defaultdict
import numpy as np
import wandb
import os, shutil, glob

def recon_loss_smaller(best_eval_result, current_eval_result):
  loss_key = 'Total_Loss'
  if not best_eval_result or loss_key not in best_eval_result:
    raise ValueError(
        'best_eval_result cannot be empty or no loss is found in it.')

  if not current_eval_result or loss_key not in current_eval_result:
    raise ValueError(
        'current_eval_result cannot be empty or no loss is found in it.')

  return best_eval_result[loss_key] > current_eval_result[loss_key]


class BestCheckpointsExporter(tf.estimator.BestExporter):
    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        if self._best_eval_result is None or \
                self._compare_fn(self._best_eval_result, eval_result):
            # copy the checkpoints files *.meta *.index, *.data* each time there is a better result, no cleanup for max amount of files here
            best_dir = os.path.join(os.path.dirname(checkpoint_path), 'best')
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            os.makedirs(best_dir)
            for name in glob.glob(checkpoint_path + '.*'):
                shutil.copy(name, os.path.join(best_dir, os.path.basename(name)))
           # also save the text file used by the estimator api to find the best checkpoint
            self._best_eval_result = eval_result
           
