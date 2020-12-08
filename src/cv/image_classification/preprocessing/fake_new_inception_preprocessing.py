from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import cv2
import tensorflow as tf
import time

def opencv_preprocess_for_train(image,height,width,begin,size):
    
  print('opeencv start')
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
  print('opeencv done')
  print(time.time())
  print(distorted_image)
  return distorted_image
  
def preprocess_for_train(image,height,width):
  with tf.name_scope(None, 'distort_image', [image, height, width]):
    #image = tf.Print(image,[tf.shape(image)])
    image = tf.Print(image,['in new.preprocess_for_train'])#
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

    height = tf.convert_to_tensor(height)
    width = tf.convert_to_tensor(width)
    #print(tf.py_func(opencv_preprocess_for_train, [image,height,width,begin,size], tf.float32))
    preprocessed_image = tf.py_func(opencv_preprocess_for_train, [image,height,width,begin,size], tf.float32,stateful=True)
    preprocessed_image = tf.Print(preprocessed_image,['in new.preprocess_for_train done'])
    #print(preprocessed_image)
    preprocessed_image = tf.Print(preprocessed_image,[preprocessed_image])
    return preprocessed_image

def preprocess_for_eval(image,height,width):
  
  def opencv_preprocess_for_eval(image,height,width):
    h,w,_ = image.shape
    croped_image = image[int(0.0625*h):int(0.9375*h),int(0.0625*w):int(0.9375*w),:]
    resized_image = cv2.resize(croped_image,(height,width),interpolation = cv2.INTER_LINEAR)
    resized_image = resized_image * 1./255.
    resized_image = (resized_image - 0.5)*2
    return resized_image
  
  image = tf.py_func(opencv_preprocess_for_eval, [image,height,width], tf.float32)
  return image
  
def preprocess_image(image,height,width,is_training=False):
  if is_training:
    #sample_distorted_bounding_box = ([10,10,0],[900,900,-1],1)
    #image = tf.Print(image,[is_training])
    return preprocess_for_train(image,height,width)
  else:
    return preprocess_for_eval(image,height,width)

