# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import codecs
import random
from PlatformNlp import utils
from PlatformNlp.data.dictionary import Dictionary
from PlatformNlp.tasks import PlatformTask, register_task

logger = logging.getLogger(__name__)


@register_task('word_embedding')
class WorkEmbeddingTask(PlatformTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--word_split', type=str, default="char", choices=["char", "word"],
                            help='build vocab ')

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.word_split in ["char", "word"], 'word_split Must be char or word'
        return WorkEmbeddingTask(args)

    def build_model(self):
        from PlatformNlp import models
        model = models.build_model(self.args, self)
        self.model = model
        return model

    def max_seq_length(self):
        return self.max_seq_length

