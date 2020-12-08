# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['TextCNNModel']
__author__ = 'xulu46'
__date__ = '2019.09.29'
"""The main textcnn model and related functions."""

import copy
import tensorflow as tf
from PlatformNlp.modules.conv_layer import conv_layer
from PlatformNlp.modules.embedding_lookup import embedding_lookup
from PlatformNlp.models import register_model, register_model_architecture
from PlatformNlp.models.platform_model import PlatformModel

@register_model('textcnn')
class TextCNNModel(PlatformModel):
    """
    ```python
    # Already been converted into WordPiece token ids
    ...
    ```
    """
    def __init__(self, features, vocab_size, embedding_size, filter_sizes, num_filters, initializer_range):
        super().__init__()

        with tf.variable_scope("textcnn"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                input_ids = features["input_ids"]
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    word_embedding_name="word_embeddings",
                    embedding_initializer=None)
                self.embedding_output = tf.expand_dims(self.embedding_output, axis=[-1])

            with tf.variable_scope("conv"):
                self.pooled_output = conv_layer(
                    input_tensor=self.embedding_output,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    initializer_range=initializer_range)

            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(self.pooled_output, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

    def get_embedding(self):
        return self.embedding_table


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--vocab_size', type=int,
                            help='vocab size')
        parser.add_argument('--embedding_size', type=int,
                            help='textcnn embedding dimension')
        parser.add_argument('--filter_sizes', type=str,
                            help='filter size for conv layer')
        parser.add_argument('--num_filters', type=int,
                            help='num filter for each filter')
        parser.add_argument('--l2_reg_lambda', type=float,
                            help='l2 reg')
        parser.add_argument('--drop_keep_prob', type=float,
                            help='dropout prob for textcnn output layer')
        parser.add_argument('--initializer_range', type=float,
                            help='initializer range for embedding')
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        filter_sizes = "2,3,4" if args.filter_sizes is None else args.filter_sizes
        filter_sizes = filter_sizes.split(",")
        filter_sizes = [int(filter_size) for filter_size in filter_sizes]
        return TextCNNModel(task.features, args.vocab_size, args.embedding_size, filter_sizes, args.num_filters, args.initializer_range)

    def get_output(self):
        return self.h_pool_flat


@register_model_architecture('textcnn', 'textcnn')
def base_architecture(args):
    args.embedding_size = 128 if args.embedding_size is None else args.embedding_size
    args.max_seq_length = 200 if args.max_seq_length is None else args.max_seq_length
    args.filter_sizes = getattr(args, 'filter_sizes', "2,3,4")
    args.num_filters =128 if args.num_filters is None else args.num_filters
    args.l2_reg_lambda = 0.1 if args.l2_reg_lambda is None else args.l2_reg_lambda
    args.drop_keep_prob = 0.9 if args.drop_keep_prob is None else args.drop_keep_prob
    args.initializer_range = 0.1 if args.initializer_range is None else args.initializer_range




