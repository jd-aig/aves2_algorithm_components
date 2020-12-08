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



# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['BertModel']
__author__ = 'xulu46'
__date__ = '2019.09.29'
"""The main bert model and related functions."""

import copy
import tensorflow as tf
from PlatformNlp.modules.embedding_lookup import embedding_postprocessor
from PlatformNlp.modules.embedding_lookup import embedding_lookup
from PlatformNlp.models import register_model, register_model_architecture
from PlatformNlp.models.platform_model import PlatformModel
from PlatformNlp.modules.utils import get_shape_list, get_activation, create_initializer
from PlatformNlp.modules.attention import create_attention_mask_from_input_mask
from PlatformNlp.modules.transformer import transformer_model


@register_model('bert')
class BertModel(PlatformModel):
    """
    ```python
    # Already been converted into WordPiece token ids
    ...
    ```
    """

    def __init__(self, features, sequence, is_training, vocab_size, hidden_size, initializer_range, type_vocab_size,
                 max_position_embeddings, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act,
                 hidden_dropout_prob, attention_probs_dropout_prob, use_one_hot_embeddings=True, scope=None):
        if "input_ids" in features:
            input_ids = features["input_ids"]
        else:
            input_ids_1 = features["input_ids_1"]
            input_ids_2 = features["input_ids_2"]
            input_ids = tf.concat([input_ids_1, input_ids_2], axis=1)

        input_mask = features["input_mask"]
        token_type_ids = features["segment_ids"]
        self.sequence = sequence
        if not is_training:
            hidden_dropout_prob = 0.0
            attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=vocab_size,
                    embedding_size=hidden_size,
                    initializer_range=initializer_range,
                    word_embedding_name="word_embeddings")

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=initializer_range,
                    max_position_embeddings=max_position_embeddings,
                    dropout_prob=hidden_dropout_prob)

                with tf.variable_scope("encoder"):
                    # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                    # mask of shape [batch_size, seq_length, seq_length] which is used
                    # for the attention scores.
                    attention_mask = create_attention_mask_from_input_mask(
                        input_ids, input_mask)

                    # Run the stacked transformer.
                    # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                    self.all_encoder_layers = transformer_model(
                        input_tensor=self.embedding_output,
                        attention_mask=attention_mask,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=intermediate_size,
                        intermediate_act_fn=get_activation(hidden_act),
                        hidden_dropout_prob=hidden_dropout_prob,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_all_layers=True)

                self.sequence_output = self.all_encoder_layers[-1]
                # The "pooler" converts the encoded sequence tensor of shape
                # [batch_size, seq_length, hidden_size] to a tensor of shape
                # [batch_size, hidden_size]. This is necessary for segment-level
                # (or segment-pair-level) classification tasks where we need a fixed
                # dimensional representation of the segment.
                with tf.variable_scope("pooler"):
                    # We "pool" the model by simply taking the hidden state corresponding
                    # to the first token. We assume that this has been pre-trained
                    first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                    self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=create_initializer(initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_output(self):
        if self.sequence == "sequence":
            return self.get_sequence_output()
        else:
            return self.get_pooled_output()

    def get_embedding(self):
        return self.embedding_table

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--vocab_size', type=int,
                            help='vocab size')
        parser.add_argument('--drop_prob', type=float,
                            help='drop out prob for output layer')
        parser.add_argument('--max_position_embeddings', type=int, default=512,
                            help='vocab size')
        parser.add_argument('--attention_probs_dropout_prob', type=float,
                            help='attention_probs_dropout_prob for each layer')
        parser.add_argument('--hidden_act', type=str, default="gelu",
                            help='hidden act')
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                            help='hidden dropout prob for each layer')
        parser.add_argument('--hidden_size', type=int, default=768,
                            help='hidden size for bert')
        parser.add_argument('--initializer_range', type=float, default=0.02,
                            help='initializer_range for bert model')
        parser.add_argument('--intermediate_size', type=int, default=3072,
                            help='intermediate_size for transformer model')
        parser.add_argument('--num_attention_heads', type=int, default=12,
                            help='num_attention_heads for transformer model')
        parser.add_argument('--num_hidden_layers', type=int, default=12,
                            help='num_hidden_layers for transformer model')
        parser.add_argument('--type_vocab_size', type=int, default=2,
                            help='type_vocab_size for transformer model')
        parser.add_argument('--l2_reg_lambda', type=float, default=0.1,
                            help='l2 reg')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if args.task == "ner":
            sequence = "sequence"
        else:
            sequence = "first"
        is_training = (task.mode == tf.estimator.ModeKeys.TRAIN)
        return BertModel(task.features, sequence, is_training, args.vocab_size, args.hidden_size,
                         args.initializer_range, args.type_vocab_size, args.max_position_embeddings,
                         args.num_hidden_layers, args.num_attention_heads, args.intermediate_size, args.hidden_act,
                         args.hidden_dropout_prob, args.attention_probs_dropout_prob)


@register_model_architecture('bert', 'bert')
def base_architecture(args):
    args.vocab_size = 21128
    args.drop_prob = 0.1 if args.drop_prob is None else args.drop_prob
    args.max_position_embeddings = 512 if args.max_position_embeddings is None else args.max_position_embeddings
    args.attention_probs_dropout_prob = 0.1 if args.attention_probs_dropout_prob is None else args.attention_probs_dropout_prob
    args.hidden_act = "gelu" if args.hidden_act is None else args.hidden_act
    args.hidden_dropout_prob = 0.1 if args.hidden_dropout_prob is None else args.hidden_dropout_prob
    args.hidden_size = 768 if args.hidden_size is None else args.hidden_size
    args.initializer_range = 0.02 if args.initializer_range is None else args.initializer_range
    args.intermediate_size = 3072 if args.intermediate_size is None else args.intermediate_size
    args.num_attention_heads = 12 if args.num_attention_heads is None else args.num_attention_heads
    args.num_hidden_layers = 12 if args.num_hidden_layers is None else args.num_hidden_layers
    args.type_vocab_size = 2 if args.type_vocab_size is None else args.type_vocab_size
    args.l2_reg_lambda = 0.1 if args.l2_reg_lambda is None else args.l2_reg_lambda



