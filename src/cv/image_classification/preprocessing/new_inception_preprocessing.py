# it is a reconsitution of inception_preprocessing with opencv tool
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
from tensorflow.python.ops import control_flow_ops
import cv2

def preprocess_for_train(image, height, width, bbox,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):

  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75,1.33),
        area_range=(0.05,1.0),
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
      begin,size,_ = sample_distorted_bounding_box
      
    if image.dtype != tf.float32:
      image = tf.cast(image, tf.float32)
      #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #image = tf.Print(image,['before preprocess',image],summarize=100)
      
    def opencv_preprocess_for_train(image,height,width,begin,size):
      #return(image)
      croped_image = image[begin[0]:begin[0]+size[0],begin[1]:begin[1]+size[1],:]
      resized_image = cv2.resize(croped_image,(height,width),interpolation = cv2.INTER_LINEAR)

      lr_flip = cv2.flip(resized_image, 1) if numpy.random.uniform()>0.5 else resized_image
      ud_flip = cv2.flip(lr_flip, 0) if numpy.random.uniform()>0.5 else lr_flip   
      #distorted_image = distort_image(ud_flip)
      alpha = numpy.random.uniform(low= -32., high= 32., size= 1 )
      blank = numpy.ones_like(ud_flip)
      adjust_brightness_image = cv2.addWeighted(ud_flip,1,blank,alpha,0)
      adjust_brightness_image[adjust_brightness_image[:,:,:]>255]=255
      adjust_brightness_image[adjust_brightness_image[:,:,:]<0]=0
      #image = cv2.inRange(image, numpy.array([0, 0, 0]), numpy.array([255, 255, 255]))
      # adjust saturation
      hsv = cv2.cvtColor(adjust_brightness_image,cv2.COLOR_RGB2HSV)
      alpha = numpy.random.uniform(low= 0.5, high= 1.5, size= 1 )
      hsv[:,:,1] = alpha * hsv[:,:,1]
      #hsv[hsv[:,:,1]>180]=180 ???
      adjust_saturation_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

      distorted_image = adjust_saturation_image * 1./255.
      distorted_image = (distorted_image - 0.5)*2
      
      return distorted_image
    
    image = tf.py_func(opencv_preprocess_for_train, [image,height,width,begin,size], tf.float32)
    #image = tf.Print(image,['after preprocess',image],summarize=100)
    return image


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    
    if image.dtype != tf.float32:
      image = tf.cast(image, tf.float32)
      #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      #image = tf.Print(image,[image])
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    def opencv_preprocess_for_eval(image,height,width):
      h,w,_ = image.shape
      #croped_image = image[int(0.0625*w):int(int(0.0625*w)+0.875*w),int(0.0625*h):int(int(0.0625*h)+0.875*h),:]
      croped_image = image[int(0.0625*h):int(int(0.0625*h)+0.875*h),int(0.0625*w):int(int(0.0625*w)+0.875*w),:]
      #croped_image = image[int(0.0625*h):int(0.9375*h),int(0.0625*w):int(0.9375*w),:]
      resized_image = cv2.resize(croped_image,(width,height),interpolation = cv2.INTER_LINEAR)
      resized_image = resized_image * 1./255.
      resized_image = (resized_image - 0.5)*2
      return resized_image
      
    height = tf.convert_to_tensor(height)
    width = tf.convert_to_tensor(width)
    image = tf.py_func(opencv_preprocess_for_eval, [image,height,width], tf.float32)
    return image

def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     add_image_summaries=True):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image. If dtype is
      tf.float32 then the range should be [0, 1], otherwise it would converted
      to tf.float32 assuming that the range is [0, MAX], where MAX is largest
      positive representable number for int(8/16/32) data type (see
      `tf.image.convert_image_dtype` for details).
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations.
    add_image_summaries: Enable image summaries.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  if is_training:
    return preprocess_for_train(image, height, width, bbox, fast_mode,
                                add_image_summaries=add_image_summaries)
  else:
    return preprocess_for_eval(image, height, width)
