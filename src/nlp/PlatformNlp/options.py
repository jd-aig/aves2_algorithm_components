import argparse
import sys
from typing import Callable, List, Optional

from PlatformNlp import utils
from PlatformNlp.data import get_available_word_split_impl, get_available_type


def get_preprocessing_parser(default_task="multi_class"):
    parser = get_parser("Preprocessing", default_task)
    add_preprocess_args(parser)
    return parser


def get_training_parser(default_task="multi_class"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser)
    return parser


def csv_str_list(x):
    return x.split(',')


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_str_dict(x, type=dict):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return x


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def parse_args_and_arch(
        parser: argparse.ArgumentParser,
        input_args: List[str] = None,
        parse_known: bool = False,
        suppress_defaults: bool = False,
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """
    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = argparse.ArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    from PlatformNlp.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        ARCH_MODEL_REGISTRY[args.arch].add_args(parser)

    # Add *-specific args to parser.
    from PlatformNlp.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)

    if hasattr(args, "task"):
        from PlatformNlp.tasks import TASK_REGISTRY
        TASK_REGISTRY[args.task].add_args(parser)

    if hasattr(args, "metrices"):
        from PlatformNlp.metrics import METRICES_REGISTRY
        METRICES_REGISTRY[args.metrices].add_args(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    # Apply architecture configuration.
    # if hasattr(args, "arch"):
    #    ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc, default_task="multi_class"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--output_dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()
    utils.import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # fmt: off
    parser.add_argument('--output_dir', metavar='DIR', default='',
                        help='path to save models')

    parser.add_argument('--model_dir', metavar='DIR', default='',
                        help='path to load models')
    from PlatformNlp.registry import REGISTRIES
    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            '--' + registry_name.replace('_', '-'),
            default=REGISTRY['default'],
            choices=REGISTRY['registry'].keys(),
        )

    # Task definitions can be found under fairseq/tasks/
    from PlatformNlp.tasks import TASK_REGISTRY
    parser.add_argument('--task', metavar='TASK', default=default_task,
                        choices=TASK_REGISTRY.keys(),
                        help='task')

    from PlatformNlp.metrics import METRICES_REGISTRY
    parser.add_argument('--metrics', metavar='METRICS', default="multi_class_cross_entry_metrics",
                        choices=METRICES_REGISTRY.keys(),
                        help='metrics')

    # fmt: on
    return parser


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    group.add_argument("--data_file", metavar="DATA",
                       help='sources data file')
    group.add_argument("--dict_file", metavar="DICT",
                       help="given dictionary or Generated object file")
    group.add_argument("--max_seq_length", metavar="MAX_SEQ_LENGTH", default=200, type=int,
                       help="max_sent_length of a sentence")
    group.add_argument("--word_format", metavar="WORD_FORMAT", choices=get_available_word_split_impl(),
                       help='choice word format to generate words')
    group.add_argument("--type", metavar="TYPE", choices=get_available_type(),
                       help="generate type")
    group.add_argument("--label_file", metavar="LABEL_FILE", help="source label file or dest label file")
    group.add_argument("--output_file", metavar="OUTPUT_FILE", help="destinate label file")
    return parser


def add_dataset_args(parser):
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    group.add_argument("--type", metavar="TYPE", choices=get_available_type(),
                       help="generate type")
    group.add_argument("--batch_size", metavar="BATCH_SIZE", type=int, help="generate type")
    group.add_argument('--data_file', metavar="DATA_FILE", type=str,
                       help='data file of the input tfrecord')
    group.add_argument('--label_file', metavar="LABEL_FILE", type=str,
                       help='label file of the input')
    group.add_argument('--train_data_file', metavar="DATA_FILE", type=str,
                       help='data file of the input tfrecord')
    group.add_argument('--eval_data_file', metavar="DATA_FILE", type=str,
                       help='data file of the input tfrecord')
    group.add_argument('--test_data_file', metavar="DATA_FILE", type=str,
                       help='data file of the input tfrecord')
    group.add_argument('--max_seq_length', metavar="MAX_SEQ_LENGTH", default=200, type=int,
                       help="max_sent_length of a sentence")

    # fmt: on
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("Optimization")
    # fmt: off
    group.add_argument('--epoch', '--e', default=1, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--clip_norm', default=0.0, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--learning_rate', default='0.001', type=float,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--warmup_proportion', default=0.1, type=float, metavar='WARMUP',
                       help='warmup_proportion')
    # fmt: on
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("Checkpointing")
    # fmt: off
    group.add_argument('--init_checkpoint', metavar='DIR', type=str,
                       help='init checkpoint name')
    group.add_argument('--device_map', metavar='DEVICE', type=str, default='-1',
                       help='devicp_map')
    group.add_argument("--save_checkpoints_steps", metavar='DEVICE', type=int, default=100,
                       help="How often to save the model checkpoint.")
    # fmt: on
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group("Interactive")
    # fmt: off
    group.add_argument('--batch_size', default=1, type=int, metavar='N',
                       help='predict n sentences each time')
    # fmt: on


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    from PlatformNlp.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='model architecture')
    group.add_argument("--do_train", dest='do_train', action='store_true', help="if do train type")
    group.add_argument("--do_eval", dest='do_eval', action='store_true', help="if do eval type")
    group.add_argument("--inter_op_parallelism_threads", type=int, default=0,
                       help="the inter_op_parallelism_threads to set the gpu config")
    group.add_argument("--intra_op_parallelism_threads", type=int, default=0,
                       help="Number of intra_op_parallelism_threads to use for CPU. ")
    group.add_argument("--max_steps_without_decrease", type=int, default=100, help="max step without decrease")

    # fmt: on
    return group
