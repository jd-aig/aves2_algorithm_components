import tensorflow as tf
from PlatformNlp.modules.utils import get_shape_list, create_initializer


def conv_layer(input_tensor,
               filter_sizes=[2, 3],
               num_filters=128,
               initializer_range=0.1):
  input_shape = get_shape_list(input_tensor, expected_rank=4)
  sequence_length = input_shape[1]
  input_width = input_shape[-2]

  pooled_outputs = []
  for i, filter_size in enumerate(filter_sizes):
      with tf.variable_scope("conv-maxpool-%s" % filter_size, default_name="conv-maxpool-0"):
          # Convolution Layer
          filter_shape = [filter_size, input_width, 1, num_filters]
          W = tf.get_variable(name="W",
                              shape=filter_shape,
                              initializer=create_initializer(initializer_range))
          b = tf.get_variable(name="b",
                              dtype = tf.float32,
                              initializer= tf.constant([0.1]*num_filters))
          conv = tf.nn.conv2d(
              input_tensor,
              W,
              strides=[1, 1, 1, 1],
              padding="VALID",
              name="conv")
          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
          # Maxpooling over the outputs
          pooled = tf.nn.max_pool(
              h,
              ksize=[1, sequence_length - filter_size + 1, 1, 1],
              strides=[1, 1, 1, 1],
              padding='VALID',
              name="pool")
          pooled_outputs.append(pooled)
  return pooled_outputs