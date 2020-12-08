# -*- coding: utf-8 -*-
"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import PIL.Image

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session

import common_flags
import datasets
import data_provider
import json
import os
import json
FLAGS = flags.FLAGS
common_flags.define()

# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path', '',
                    'A file pattern with a placeholder for the image index.')
flags.DEFINE_string('result_path', '',
                    'A file pattern with a placeholder for the image index.')

def get_dataset_image_size(dataset_name):
  # Ideally this info should be exposed through the dataset interface itself.
  # But currently it is not available by other means.
  ds_module = getattr(datasets, dataset_name)
  config_path = os.path.join(FLAGS.config_dir,'newdataset_config_json.json')  
  config = json.load(open(config_path,'r'))
  height, width, _ = config['image_shape']
  return width, height

def load_images(path,dataset_name):
  filenames = []
  width, height = get_dataset_image_size(dataset_name)
  images_actual_data = []
  #images_actual_data = np.ndarray(shape=(1, height, width, 3),
  #                                dtype='uint8')
  for i in tf.gfile.Glob(path+'*.jpg'):
    print("Reading %s" % i)
    pil_image = PIL.Image.open(i)
    pil_image = pil_image.resize((width, height),PIL.Image.ANTIALIAS)
    images_actual_data.append(np.asarray(pil_image))
    filenames.append(i)
  return images_actual_data,filenames,len(images_actual_data)

def create_model(batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  dataset,_ = common_flags.create_dataset('train')
  model = common_flags.create_model(
    num_char_classes=dataset.num_char_classes,
    seq_length=dataset.max_sequence_length,
    num_views=dataset.num_of_views,
    null_code=dataset.null_code,
    charset=dataset.charset)
  raw_images = tf.placeholder(tf.uint8, shape=[batch_size, height, width, 3])
  images = tf.map_fn(data_provider.preprocess_image, raw_images,
                     dtype=tf.float32)
  endpoints = model.create_base(images, labels_one_hot=None)
  return raw_images, endpoints


def run(checkpoint, batch_size, dataset_name, image_path_pattern):
  result = {}
  images_data, image_files, num_images = load_images(image_path_pattern,dataset_name)
  
  images_placeholder, endpoints = create_model(1,
                                               dataset_name)
  session_creator = monitored_session.ChiefSessionCreator(
    checkpoint_filename_with_path=checkpoint)
  with monitored_session.MonitoredSession(
      session_creator=session_creator) as sess:
    for i in range(num_images):
        image = images_data[i]
        image = image[np.newaxis,:,:,:]
        predictions = sess.run(endpoints.predicted_text,feed_dict={images_placeholder: image})
        print("image {} is predicted as {}".format(image_files[i],predictions[0]))
        result[image_files[i]] = predictions[0]
  result_json = os.path.join(FLAGS.result_path,'predict_result.json')
  json.dump(result,open(result_json,'w'),indent=4)
  return predictions


def main(_):
  print("Predicted strings:")
  checkpoint = tf.train.latest_checkpoint(
    FLAGS.train_log_dir,
    latest_filename=None)
  print(checkpoint)
  if not os.path.exists(FLAGS.result_path): 
      os.mkdir(FLAGS.result_path)
  predictions = run(checkpoint, FLAGS.batch_size, FLAGS.dataset_name,
                  FLAGS.image_path)


if __name__ == '__main__':
  tf.app.run()
