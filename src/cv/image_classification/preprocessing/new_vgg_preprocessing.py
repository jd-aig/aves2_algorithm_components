# it is a reconsitution of vgg_preprocessing with opencv
# ==============================================================================
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import cv2
slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  
  def opencv_preprocess_for_train(image,height,width):
    h,w,_ = image.shape
    resize_side = numpy.random.randint(low=resize_side_min, high=resize_side_max+1,size=1)
    if h < w :
      resize_h = resize_side
      resize_w = w*resize_side/h
    else:
      resize_w = resize_side
      resize_h = h*resize_side/w
    resized_image  = cv2.resize(image,(int(resize_w),int(resize_h)),interpolation = cv2.INTER_LINEAR)
    # w,h
    #print(resized_image.shape)
    begin_h = numpy.random.randint(low=0, high=resize_h-height, size=1)
    begin_w = numpy.random.randint(low=0, high=resize_w-width, size=1)
    #croped_image = resized_image[0:100,100:200,:]
    croped_image = resized_image[int(begin_h):int(begin_h+height),int(begin_w):int(begin_w+width),:]
    lr_flip = cv2.flip(croped_image, 1) if numpy.random.uniform()>0.5 else croped_image
    lr_flip[:,:,0] = lr_flip[:,:,0]-123.68 #r
    lr_flip[:,:,1] = lr_flip[:,:,1]-116.78 #g
    lr_flip[:,:,2] = lr_flip[:,:,2]-103.94 #b
    return lr_flip
  image = tf.py_func(opencv_preprocess_for_train, [image,output_height,output_width], tf.float32)
  #image = tf.Print(image,['after preprocess',image],summarize=100)
  return image

def preprocess_for_eval(image, output_height, output_width, resize_side_vgg):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  def opencv_preprocess_for_eval(image,height,width):
    h,w,_ = image.shape
    resize_side = resize_side_vgg
    if h < w :
      resize_h = resize_side
      resize_w = w*resize_side/h
    else:
      resize_w = resize_side
      resize_h = h*resize_side/w
      
    resized_image  = cv2.resize(image,(int(resize_w),int(resize_h)),interpolation = cv2.INTER_LINEAR)
    #print(resized_image.shape)
    begin_h = int((resize_h-height)*0.5)
    begin_w = int((resize_w-width)*0.5)
    croped_image = resized_image[begin_h:int(begin_h+height),begin_w:int(begin_w+width),:]
    croped_image[:,:,0] = croped_image[:,:,0]-123.68 #r
    croped_image[:,:,1] = croped_image[:,:,1]-116.78 #g
    croped_image[:,:,2] = croped_image[:,:,2]-103.94 #b
    #print(croped_image.shape)
    return croped_image

  output_height = tf.convert_to_tensor(output_height)
  output_width = tf.convert_to_tensor(output_width)
  image = tf.py_func(opencv_preprocess_for_eval, [image,output_height,output_width], tf.float32)
  return image

def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                                resize_side_min, resize_side_max)
  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)
