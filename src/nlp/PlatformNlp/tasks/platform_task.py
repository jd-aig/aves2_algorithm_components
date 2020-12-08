import logging
import os
import warnings

from PlatformNlp.data.dictionary import Dictionary

logger = logging.getLogger(__name__)


class PlatformTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, args):
        self.args = args

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, tokenizer):
        d = Dictionary()
        for filename in filenames:
            counter = Dictionary.add_file_to_dictionary(filename, tokenizer)
            for w, c in sorted(counter.items()):
                d.add_symbol(w, c)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    @classmethod
    def load_dataset(cls, args):
        """Load a given dataset split (e.g., train, valid, test)."""
        from PlatformNlp import data
        dataset = data.get_dataset(args)
        if args.type == "train":
            is_training = True
        else:
            is_training = False
        input_fn = dataset.builder(args.data_file, is_training, args.batch_size, is_training, args)
        return input_fn

    def build_criterion(self, args):
        """
        Build the :class:`~PlatformNlp.criterions.PlatformNlpCriterion` instance for
        this task.
        """
        from PlatformNlp import criterions
        return criterions.build_criterion(args, self)

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from PlatformNlp import models
        model = models.build_model(args, self)
        self.model = model
        return model

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from PlatformNlp import criterions

        return criterions.build_criterion(args, self)


    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~PlatformNlp.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError


