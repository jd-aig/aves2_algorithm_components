# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['CdssmModel']
__author__ = 'xulu46'
__date__ = '2020.10.14'
"""The main cdssm model and related functions."""

import copy
import tensorflow as tf
from PlatformNlp.modules.embedding_lookup import embedding_lookup
from PlatformNlp.models import register_model, register_model_architecture
from PlatformNlp.models.platform_model import PlatformModel
from PlatformNlp.modules.dssm_layer import dssm_layer
from PlatformNlp.modules.utils import get_activation
from PlatformNlp.modules.conv_layer import conv_layer


@register_model('cdssm')
class CdssmModel(PlatformModel):
    """
    ```python
    # Already been converted into WordPiece token ids
    ...
    ```
    """
    def __init__(self, is_training, features, vocab_size, filter_sizes, num_filters, act, embedding_size, hidden_sizes, max_seq_length, dropout_prob, initializer_range):
        query_ids = features["input_ids_1"]
        doc_ids = features["input_ids_2"]
        with tf.variable_scope("cdssm"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.query_embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=query_ids,
                    vocab_size=vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    word_embedding_name="query_embeddings",
                    embedding_initializer=None)
                self.query_embedding_output = tf.expand_dims(self.query_embedding_output, axis=[-1])
                (self.doc_embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=doc_ids,
                    vocab_size=vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    word_embedding_name="doc_embeddings",
                    embedding_initializer=None)
                self.doc_embedding_output = tf.expand_dims(self.doc_embedding_output, axis=[-1])

                with tf.variable_scope("conv"):
                    self.query_conv_output = conv_layer(
                        input_tensor=self.query_embedding_output,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        initializer_range=initializer_range)

                    num_filters_total = num_filters * len(filter_sizes)
                    self.query_h_pool = tf.concat(self.query_conv_output, 3)
                    self.query_h_pool_flat = tf.reshape(self.query_h_pool, [-1, num_filters_total])

                    self.doc_conv_output = conv_layer(
                        input_tensor=self.doc_embedding_output,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        initializer_range=initializer_range)

                    self.doc_h_pool = tf.concat(self.doc_conv_output, 3)
                    self.doc_h_pool_flat = tf.reshape(self.doc_h_pool, [-1, num_filters_total])

                with tf.variable_scope("dssm"):
                    self.query_pred, self.doc_pred, self.cos_sim_prob = dssm_layer(self.query_h_pool_flat, self.doc_h_pool_flat, hidden_sizes, get_activation(act), is_training, max_seq_length, embedding_size, initializer_range, dropout_prob)

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
        parser.add_argument('--act', type=str, choices=["tanh", "linear", "relu", "gelu"],
                            help='filter size for conv layer')
        parser.add_argument('--hidden_sizes', type=str,
                            help='num filter for each filter')
        parser.add_argument('--filter_sizes', type=str,
                            help='filter size for conv layer')
        parser.add_argument('--num_filters', type=int,
                            help='num filter for each filter')
        parser.add_argument('--l2_reg_lambda', type=float,
                            help='l2 reg')
        parser.add_argument('--drop_prob', type=float,
                            help='dropout prob for textcnn output layer')
        parser.add_argument('--initializer_range', type=float,
                            help='initializer range for embedding')
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        hidden_sizes = "500, 200" if args.hidden_sizes is None else args.hidden_sizes
        hidden_sizes = hidden_sizes.split(",")
        hidden_sizes = [int(hidden_size) for hidden_size in hidden_sizes]
        filter_sizes = "2,3,4" if args.filter_sizes is None else args.filter_sizes
        filter_sizes = filter_sizes.split(",")
        filter_sizes = [int(filter_size) for filter_size in filter_sizes]
        istraining = task.mode == tf.estimator.ModeKeys.TRAIN
        return CdssmModel(istraining, task.features, args.vocab_size, filter_sizes, args.num_filters, args.act, args.embedding_size, hidden_sizes, args.max_seq_length, args.drop_prob, args.initializer_range)

    def get_output(self):
        return self.cos_sim_prob


@register_model_architecture('cdssm', 'cdssm')
def base_architecture(args):
    args.vocab_size = 21128 if args.vocab_size is None else args.vocab_size
    args.act = "relu" if args.act is None else args.act
    args.embedding_size = 128 if args.embedding_size is None else args.embedding_size
    args.max_seq_length = 200 if args.max_seq_length is None else args.max_seq_length
    args.hidden_sizes = getattr(args, 'hidden_sizes', "500,200")
    args.num_filters =128 if args.num_filters is None else args.num_filters
    args.filter_sizes = getattr(args, 'filter_sizes', "2,3,4")
    args.num_filters = 128 if args.num_filters is None else args.num_filters
    args.l2_reg_lambda = 0.1 if args.l2_reg_lambda is None else args.l2_reg_lambda
    args.drop_prob = 0.9 if args.drop_prob is None else args.drop_prob
    args.initializer_range = 0.1 if args.initializer_range is None else args.initializer_range
