# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to evaluate a trained Attention OCR model.

A simple usage example:
python eval.py
"""
import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
import data_provider
import common_flags
import collections

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable

flags.DEFINE_string('eval_log_dir', '/tmp/attention_ocr/eval',
                    'Directory where the evaluation results are saved to.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency in seconds to run evaluations.')

flags.DEFINE_integer('number_of_steps', 1,
                     'Number of times to run evaluation.')
# yapf: enable

def main(_):
  if not tf.gfile.Exists(FLAGS.eval_log_dir):
    tf.gfile.MakeDirs(FLAGS.eval_log_dir)
  
  dataset,num_examples = common_flags.create_dataset(split_name='test')
  model = common_flags.create_model(dataset.num_char_classes,
                                    dataset.max_sequence_length,
                                    dataset.num_of_views, dataset.null_code)
  batch_size = FLAGS.batch_size if int(num_examples/FLAGS.batch_size) > 0 else num_examples 
  num_batches = int(num_examples/batch_size)
  ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'])
  DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=1, min_after_dequeue=0)
  data = data_provider.get_data(
      dataset,
      batch_size,
      augment=False,
      shuffle_config = DEFAULT_SHUFFLE_CONFIG,
      central_crop_size=common_flags.get_crop_size(),
      shuffle=False)

  endpoints = model.create_base(data.images, labels_one_hot=None)
  model.create_loss(data, endpoints)
  eval_ops = model.create_summaries(
      data, endpoints, dataset.charset, is_training=False)
  slim.get_or_create_global_step()
  session_config = tf.ConfigProto(device_count={"GPU": 0})
  """
  slim.evaluation.evaluate_once(
      master=FLAGS.master,
      checkpoint_path=FLAGS.train_log_di,
      logdir=FLAGS.eval_log_dir,
      num_evals=1,
      eval_op=eval_ops)
  """ 
  slim.evaluation.evaluation_loop(
      master=FLAGS.master,
      checkpoint_dir=FLAGS.train_log_dir,
      logdir=FLAGS.eval_log_dir,
      eval_op=eval_ops,
      num_evals=num_batches,
      eval_interval_secs=FLAGS.eval_interval_secs,
      #eval_interval_secs=0,
      max_number_of_evaluations=FLAGS.number_of_steps,
      session_config=session_config)

if __name__ == '__main__':
  app.run()
