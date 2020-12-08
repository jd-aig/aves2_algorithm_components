import tensorflow as tf
import math

def parse_example_proto(example_serialized):
  # a dict mapping from feature keys to tensor and sparsetensor values
  feature_map = {
      'image/height': tf.VarLenFeature(dtype=tf.int64),
      'image/width': tf.VarLenFeature(dtype=tf.int64),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      #'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      #'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
      'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
  }

  
  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  return features['image/encoded'], label, features['image/filename'] #, features['image/class/text']

def preprocess(image_buffer, is_training, model_name, height, width):
  
  image = tf.image.decode_image(image_buffer, channels=3)
  image = tf.cast(image, tf.float32)
  #image = tf.image.decode_image(image_buffer, channels=3, dtype=tf.float32)
  #print(image)
  pred = tf.equal(is_training,'train')
 
  from preprocessing import preprocessing_factory
  def f1(): 
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(model_name,is_training=True)
    images = image_preprocessing_fn(image, height, width)
    #images = tf.Print(images,[model_name])
    return images
  def f2(): 
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(model_name,is_training=False)
    images = image_preprocessing_fn(image, height, width)
    #images = tf.Print(images,[model_name])
    return images
  imagess = tf.cond(pred, f1,  f2)
  #imagess = tf.Print(imagess,[imagess])
  return imagess
  #preprocessing_factory.get_preprocessing(model_name,is_training=True)
  #preprocessing_factory.get_preprocessing(model_name,is_training=False)