# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['Word2vecModel']
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


@register_model('word2vec')
class Word2vecModel(PlatformModel):
    """
    ```python
    # Already been converted into WordPiece token ids
    ...
    ```
    """
    def __init__(self, features, vocab_size, embedding_size, initializer_range):
        super().__init__()
        input_ids = features["input_ids"]
        input_ids = tf.sparse_tensor_to_dense(input_ids)
        with tf.variable_scope("word2vec"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    word_embedding_name="embeddings",
                    embedding_initializer=None)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--vocab_size', type=int,
                            help='vocab size')
        parser.add_argument('--embedding_size', type=int,
                            help='textcnn embedding dimension')
        parser.add_argument('--num_sampled', type=int,
                            help='num sampled for negative sampling')
        parser.add_argument('--min_count', type=int,
                            help='min count for counting')
        parser.add_argument('--skip_window', type=int,
                            help='skip window for training')
        parser.add_argument('--num_skips', type=int,
                            help='num_skips for training')
        parser.add_argument('--initializer_range', type=float,
                            help='initializer range for embedding')
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        return Word2vecModel(task.features, args.vocab_size, args.embedding_size, args.initializer_range)

    def get_output(self):
        return self.embedding_output

    def get_embedding(self):
        return self.embedding_table


@register_model_architecture('word2vec', 'word2vec')
def base_architecture(args):
    args.vocab_size = 21128 if args.vocab_size is None else args.vocab_size
    args.embedding_size = 128 if args.embedding_size is None else args.embedding_size
    args.num_sampled = 64 if args.num_sampled is None else args.num_sampled
    args.min_count = 5 if args.min_count is None else args.min_count
    args.skip_window = 2 if args.skip_window is None else args.skip_window
    args.num_skips = 0.1 if args.num_skips is None else args.num_skips
    args.l2_reg_lambda = 0.9 if args.l2_reg_lambda is None else args.l2_reg_lambda
    args.initializer_range = 0.1 if args.initializer_range is None else args.initializer_range




