

class PlatformNlpCriterion():
    def __init__(self, args, task):
        self.args = args
        self.task = task

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""

        return cls(args, task)
