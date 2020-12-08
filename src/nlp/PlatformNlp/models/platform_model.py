"""
Base classes for various platform models.
"""

import logging

logger = logging.getLogger(__name__)

class PlatformModel():
    """Base class for Platform models."""

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")
