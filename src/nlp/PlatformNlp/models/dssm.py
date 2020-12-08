# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['DssmModel']
__author__ = 'xulu46'
__date__ = '2020.10.14'
"""The main dssm model and related functions."""

import copy
import tensorflow as tf
from PlatformNlp.modules.embedding_lookup import embedding_lookup
from PlatformNlp.models import register_model, register_model_architecture
from PlatformNlp.models.platform_model import PlatformModel
from PlatformNlp.modules.dssm_layer import dssm_layer
from PlatformNlp.modules.utils import get_activation


@register_model('dssm')
class DssmModel(PlatformModel):
    """
    ```python
    # Already been converted into WordPiece token ids
    ...
    ```
    """

    def __init__(self, is_training, features, vocab_size, act, embedding_size, hidden_sizes, max_seq_length,
                 dropout_prob, initializer_range):
        query_ids = features["input_ids_1"]
        doc_ids = features["input_ids_2"]
        with tf.variable_scope("dssm"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.query_embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=query_ids,
                    vocab_size=vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    word_embedding_name="query_embeddings",
                    embedding_initializer=None)
                (self.doc_embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=doc_ids,
                    vocab_size=vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    word_embedding_name="doc_embeddings",
                    embedding_initializer=None)

            with tf.variable_scope("dssm"):
                self.query_pred, self.doc_pred, self.cos_sim_prob = dssm_layer(self.query_embedding_output,
                                                                               self.doc_embedding_output, hidden_sizes,
                                                                               get_activation(act), is_training,
                                                                               max_seq_length, embedding_size,
                                                                               initializer_range, dropout_prob)

    def get_embedding(self):
        return self.embedding_table

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--vocab_size', type=int,
                            help='vocab size')
        parser.add_argument('--neg', type=int,
                            help="neg sampling")
        parser.add_argument('--embedding_size', type=int,
                            help='textcnn embedding dimension')
        parser.add_argument('--act', type=str, choices=["tanh", "linear", "relu", "gelu"],
                            help='filter size for conv layer')
        parser.add_argument('--hidden_sizes', type=str,
                            help='num filter for each filter')
        parser.add_argument('--l2_reg_lambda', type=float,
                            help='l2 reg')
        parser.add_argument('--drop_prob', type=float,
                            help='dropout prob for textcnn output layer')
        parser.add_argument('--initializer_range', type=float, default=0.1,
                            help='initializer range for embedding')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        hidden_sizes = "500, 200" if args.hidden_sizes is None else args.hidden_sizes
        hidden_sizes = hidden_sizes.split(",")
        hidden_sizes = [int(hidden_size) for hidden_size in hidden_sizes]
        istraining = task.mode == tf.estimator.ModeKeys.TRAIN
        return DssmModel(istraining, task.features, args.vocab_size, args.act, args.embedding_size, hidden_sizes,
                         args.max_seq_length, args.drop_prob, args.initializer_range)

    def get_output(self):
        return self.cos_sim_prob


@register_model_architecture('dssm', 'dssm')
def base_architecture(args):
    args.vocab_size = 21128 if args.vocab_size is None else args.vocab_size
    args.neg = 4 if args.neg is None else args.neg
    args.act = "relu" if args.act is None else args.act
    args.embedding_size = 128 if args.embedding_size is None else args.embedding_size
    args.max_seq_length = 200 if args.max_seq_length is None else args.max_seq_length
    args.l2_reg_lambda = 0.1 if args.l2_reg_lambda is None else args.l2_reg_lambda
    args.hidden_sizes = getattr(args, 'hidden_sizes', "500,200")
    args.drop_prob = 0.1 if args.drop_prob is None else args.drop_prob





