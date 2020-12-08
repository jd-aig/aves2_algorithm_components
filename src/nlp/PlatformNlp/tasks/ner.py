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


@register_task('ner')
class NerTask(PlatformTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--num_classes', type=int,
                            help='build vocab ')

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num_classes'
        return NerTask(args)

    def build_model(self):
        from PlatformNlp import models
        model = models.build_model(self.args, self)
        self.model = model
        return model

    def max_seq_length(self):
        return self.max_seq_length

