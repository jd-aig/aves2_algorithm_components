import logging
import os

import numpy as np
import codecs
import random
from PlatformNlp import utils
from PlatformNlp.data.dictionary import Dictionary
from PlatformNlp.tasks import PlatformTask, register_task

import tensorflow as tf


logger = logging.getLogger(__name__)


@register_task('text_similarity')
class TextSimilarityTask(PlatformTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--num_classes', type=int, default=2,
                            help='number of classes or regression targets')

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num_classes'
        return TextSimilarityTask(args)

    def build_model(self):
        from PlatformNlp import models
        model = models.build_model(self.args, self)
        self.model = model
        return model

    def max_seq_length(self):
        return self.max_seq_length

