# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import os

from .platform_metrics import PlatformMetrice

METRICES_REGISTRY = {}
METRICES_CLASS_NAMES = set()


def register_metrices(name):
    """
    New metrices can be added to PlatformNlp with the
    :func:`~PlatformNlp.metrics.register_metrices` function decorator.
    """

    def register_metrices_cls(cls):
        if name in METRICES_REGISTRY:
            raise ValueError('Cannot register duplicate metrices ({})'.format(name))
        if not issubclass(cls, PlatformMetrice):
            raise ValueError('metrices ({}: {}) must extend PlatformTask'.format(name, cls.__name__))
        if cls.__name__ in METRICES_CLASS_NAMES:
            raise ValueError('Cannot register metrices with duplicate class name ({})'.format(cls.__name__))
        METRICES_REGISTRY[name] = cls
        METRICES_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_metrices_cls


def get_metrices(name):
    return METRICES_REGISTRY[name]


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        task_name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('PlatformNlp.metrics.' + task_name)