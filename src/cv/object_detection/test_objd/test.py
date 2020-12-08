from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'data_dir', None, 'Path to data directory '
    'where event and checkpoint files will be written.')
tf.app.flags.DEFINE_string(
    'output_path', None, 'Path to output data directory '
    'where event and checkpoint files will be written.')
tf.app.flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
tf.app.flags.DEFINE_float(
    'min_score_thresh', 0.5, 'min_score_thresh')


PATH_TO_FROZEN_GRAPH = os.path.join(FLAGS.model_dir,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(FLAGS.data_dir,'ImageSets/label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = os.path.join(FLAGS.data_dir,'JPEGImages')
VAL_TXT_DIR = os.path.join(FLAGS.data_dir,'ImageSets')
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg')]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

with detection_graph.as_default():
  with tf.Session() as sess:
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # Run inference

    VAL_TXT = os.path.join(VAL_TXT_DIR,'val.txt')
    with open(VAL_TXT,'r') as f:
      content = [line.strip() for line in f]
      #print(content)

    for image_name in content:
      TEST_IMAGE_PATHS = None
      ext = None
      if os.path.exists(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.jpg')):
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.jpg')
        ext = '.jpg'
      elif os.path.exists(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.JPG')):
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.JPG')
        ext = '.JPG'
      elif os.path.exists(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.jpeg')):
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.jpeg')
        ext = '.jpeg'
      elif os.path.exists(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.JPEG')):
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.JPEG')
        ext = '.JPEG'
      elif os.path.exists(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.png')):
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.png')
        ext = '.png'
      elif os.path.exists(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.PNG')):
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name+'.PNG')
        ext = '.PNG'
      else:
        print(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_name),' is not exists!')
        continue
      #print(image_name)
      print(TEST_IMAGE_PATHS)
      image = Image.open(TEST_IMAGE_PATHS)
      #image = Image.open("/export/luozhuang/data_tf/tfserving/client_server/images/1.jpg")
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      start = time.time()
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image_np_expanded}) 
      print("per image cost time=%.2f ms" %(1000*(time.time()-start)))

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)

      #print(output_dict['detection_classes'])
      #break
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

      vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        min_score_thresh=FLAGS.min_score_thresh,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
      im = Image.fromarray(image_np)
      im.save(os.path.join(FLAGS.output_path,image_name+ext))
