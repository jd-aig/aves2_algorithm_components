import tensorflow as tf
import json
import os

from model import model_fn
from src.input_pipeline import Pipeline
import argparse
tf.logging.set_verbosity('INFO')
parser = argparse.ArgumentParser()
parser.add_argument("--train_tfrecord", type=str, default="../data/")
parser.add_argument("--val_tfrecord", type=str, default="../data/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

CONFIG = 'config.json'
GPU_TO_USE = '0'

params = json.load(open(CONFIG))
model_params = params['model_params']
input_params = params['input_pipeline_params']


def get_input_fn(is_training=True):

    image_size = input_params['image_size'] if is_training else None
    # (for evaluation i use images of different sizes)
    dataset_path = args.train_tfrecord if is_training else args.val_tfrecord
    batch_size = args.batch_size if is_training else 1
    # for evaluation it's important to set batch_size to 1

    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            features, labels = pipeline.get_batch()
        return features, labels

    return input_fn


config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=args.output_path,
    session_config=config,
    save_summary_steps=200,
    save_checkpoints_secs=600,
    log_step_count_steps=100,
    keep_checkpoint_max=1
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)

fid = open(os.path.join(args.train_tfrecord,'num_examples.txt'),'r')
num_examples = int(fid.read())
print('num_examples : ',num_examples)
num_steps = (num_examples*args.epochs)//args.batch_size
if num_steps == 0:
  num_steps = 1
fid.close()

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=num_steps)
eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=None, start_delay_secs=1800, throttle_secs=1800)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
