import logging
import os
import sys

sys.path.append("../")
sys.path.append("../PlatformNlp/")
from PlatformNlp import options, tasks, utils
from PlatformNlp.data import get_dataset
from PlatformNlp.tokenization import CharTokenizer, WordTokenizer

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('PlatformNlp.preprocess')


def main(args):
    utils.import_user_module(args)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.addHandler(logging.FileHandler(
        filename=os.path.join(args.output_dir, 'preprocess.log'),
    ))
    logger.info(args)

    # first create task
    task = tasks.get_task(args.task)

    # create dict
    if args.dict_file is not None and os.path.exists(args.dict_file):
        dict = task.load_dictionary(args.dict_file)
    else:
        dict = None

    # create tokenizer
    if args.word_format == "char":
        tokenizer = CharTokenizer(dict)
    else:
        tokenizer = WordTokenizer(dict)

    if not os.path.exists(args.dict_file):
        # build dict and reload dict
        dict = task.build_dictionary(args.data_file, tokenizer)
        tokenizer.set_dict(dict)
    d = get_dataset(args)
    d.build_dataset(args, tokenizer)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = options.parse_args_and_arch(parser, modify_parser=None)

    main(args)


if __name__ == "__main__":
    cli_main()
