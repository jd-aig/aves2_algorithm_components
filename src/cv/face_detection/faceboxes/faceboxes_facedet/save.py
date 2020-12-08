import tensorflow as tf
import json
from model import model_fn
import argparse
import os

"""The purpose of this script is to export a savedmodel."""
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="./output/")
args = parser.parse_args()

CONFIG = 'config.json'
OUTPUT_FOLDER = os.path.join(args.output_path,'tmp')
GPU_TO_USE = '0'

WIDTH, HEIGHT = None, None
# size of an input image,
# set (None, None) if you want inference
# for images of variable size


tf.logging.set_verbosity('INFO')
params = json.load(open(CONFIG))
model_params = params['model_params']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=args.output_path,
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.uint8, shape=[None, HEIGHT, WIDTH, 3], name='image_tensor')
    features = {'images': tf.to_float(images)*(1.0/255.0)}
    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)
