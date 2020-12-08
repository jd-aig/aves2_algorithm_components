import argparse
import logging
import time
import sys
import os
import shutil

sys.path.append("../")
sys.path.append("../PlatformNlp/")
from PlatformNlp.utils import total_sample

import tensorflow as tf
from PlatformNlp import (
    checkpoint_util,
    options,
    tasks,
    utils,
)
from PlatformNlp.optim.AdamWeightDecayOptimizer import create_optimizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("Platform_cli.train")


def model_fn_builder(args, task, num_train_steps, num_warm_up_steps):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        task.features = features
        label_id = features["label_ids"]
        task.labels = label_id
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_id)[0], dtype=tf.float32)

        from PlatformNlp import models
        task.mode = mode
        task.build_model()
        print("build model done")

        criterion = task.build_criterion(args)
        (total_loss, per_example_loss, logits, probabilities) = criterion.get_loss()
        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        if args.init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = checkpoint_util.get_assignment_map_from_checkpoint(tvars, args.init_checkpoint)

            tf.train.init_from_checkpoint(args.init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        predictions = {"output": probabilities}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = args.learning_rate
            train_op = create_optimizer(total_loss, learning_rate, num_train_steps, num_warm_up_steps)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                export_outputs=export_outputs)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = metric_fn(per_example_loss, label_id, logits, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                export_outputs=export_outputs)
        return output_spec

    return model_fn


def main(args):
    utils.import_user_module(args)
    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # Build model and criterion

    tf.logging.set_verbosity(tf.logging.INFO)
    start = time.time()
    if args.model_dir is not None and args.init_checkpoint is not None:
        args.init_checkpoint = os.path.join(args.model_dir, args.init_checkpoint)
    else:
        args.init_checkpoint = None

    if args.output_dir is not None and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    eval_dir = os.path.join(args.output_dir, "eval")
    if eval_dir is not None and not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    if args.max_seq_length > 520:
        raise ValueError(
            "Cannot use sequence length %d because the textcnn model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, 520))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=args.inter_op_parallelism_threads,
        intra_op_parallelism_threads=args.intra_op_parallelism_threads,
        allow_soft_placement=True)
    log_every_n_steps = 8
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=log_every_n_steps,
        save_checkpoints_steps=args.save_checkpoints_steps,
        session_config=session_config,
        model_dir=args.output_dir)

    num_examples = total_sample(args.train_data_file)
    num_train_steps = int(num_examples / int(args.batch_size) * int(args.epoch))
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    model_fn = model_fn_builder(args, task, num_train_steps, num_warmup_steps)

    estimator = None
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.output_dir,
        params={
            'batch_size': int(args.batch_size),
            # 'embedding_initializer': embedding_matrix,
        },
        config=run_config)

    args.data_file = args.train_data_file
    train_input_fn = tasks.PlatformTask.load_dataset(args)
    if args.do_eval:
        args.data_file = args.eval_data_file
        eval_input_fn = tasks.PlatformTask.load_dataset(args)
        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='eval_loss',
            max_steps_without_decrease=int(args.max_steps_without_decrease),
            min_steps=10,
            run_every_secs=None,
            run_every_steps=int(args.save_checkpoints_steps))
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec=tf.estimator.TrainSpec(train_input_fn, max_steps=num_train_steps,
                                              hooks=[early_stopping_hook]),
            eval_spec=tf.estimator.EvalSpec(eval_input_fn, steps=100))
    else:
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("whole time: {}s".format(time.time() - start))


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    main(args)


if __name__ == "__main__":
    cli_main()
