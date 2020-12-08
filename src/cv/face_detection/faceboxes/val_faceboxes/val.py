import tensorflow as tf
import json
import os

from model import model_fn
from src.input_pipeline import Pipeline
import argparse
tf.logging.set_verbosity('INFO')
parser = argparse.ArgumentParser()
parser.add_argument("--val_tfrecord", type=str, default="../data/")
parser.add_argument("--model_dir", type=str, default="./output/")
parser.add_argument("--output_path", type=str, default="./output/")
args = parser.parse_args()

CONFIG = 'config.json'
GPU_TO_USE = '0'

params = json.load(open(CONFIG))
model_params = params['model_params']
input_params = params['input_pipeline_params']


def get_input_fn(is_training=True):

    image_size = input_params['image_size'] if is_training else None
    # (for evaluation i use images of different sizes)
    dataset_path = args.val_tfrecord
    batch_size = 1
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
    model_dir=args.model_dir,
    session_config=config,
    save_summary_steps=200,
    save_checkpoints_secs=600,
    log_step_count_steps=100,
    keep_checkpoint_max=1
)

val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)

eval_metrics = estimator.evaluate(input_fn=val_input_fn,checkpoint_path=os.path.join(args.model_dir,'model.ckpt'))
print(eval_metrics)

json_dict={}
json_dict["recall"] = str(eval_metrics['metrics/recall'])
json_dict["precision"] = str(eval_metrics['metrics/precision'])
json_dict["AP"] = str(eval_metrics['metrics/AP'])
json_data = json.dumps(json_dict)
f=open(os.path.join(args.output_path,"result.json"),"w")
f.write(str(json_data))
f.close()
