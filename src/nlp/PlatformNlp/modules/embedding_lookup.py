# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




import tensorflow as tf
from PlatformNlp.modules.utils import create_initializer
from PlatformNlp.modules.utils import get_shape_list
from PlatformNlp.modules.layer_norm_and_drop_out import layer_norm_and_dropout


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     embedding_initializer=None):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length].
    if embedding_initializer is not None:
        embedding_initializer = tf.constant_initializer(embedding_initializer)

    trainable = True
    if embedding_initializer is not None:
        trainable = False
    else:
        trainable = True
    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=embedding_initializer if embedding_initializer is not None else create_initializer(
            initializer_range),
        trainable=trainable)

    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


