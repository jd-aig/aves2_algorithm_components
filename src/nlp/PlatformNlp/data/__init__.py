import argparse
import importlib
import os
from PlatformNlp.data.base_dataset import BaseDataset

DATA_REGISTRY = {}


def get_available_word_split_impl():
    return ['word', 'char']


def get_available_type():
    return ["train", "valid", "test"]


def get_dataset(args):
    return DATA_REGISTRY[args.task](args)


def register_dataset(name):
    """
    New dataset types can be added to platform with the :func:`register_data`
    function decorator.

    For example::

        @register_dataset('multi_class')
        class MultiClassFixLenDataset():
            (...)

    .. note:: All datasets must implement the :class:`BaseDataset` interface.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in DATA_REGISTRY:
            raise ValueError('Cannot register duplicate dataset ({})'.format(name))
        if not issubclass(cls, BaseDataset):
            raise ValueError('Dataset ({}: {}) must extend BaseDataset'.format(name, cls.__name__))
        DATA_REGISTRY[name] = cls
        return cls

    return register_model_cls

# automatically import any Python files in the data/ directory
datasets_dir = os.path.dirname(__file__)
for file in os.listdir(datasets_dir):
    path = os.path.join(datasets_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('PlatformNlp.data.' + model_name)

