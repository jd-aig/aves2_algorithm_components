import logging
import os
import warnings

logger = logging.getLogger(__name__)


class PlatformMetrice(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, args, input_ids, label_ids, predict_scores, label_mapping):
        self.args = args
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.predict_scores = predict_scores
        self.label_mapping = label_mapping
