import math
import tensorflow as tf


from PlatformNlp.criterions.platform_criterion import PlatformNlpCriterion
from PlatformNlp.criterions import register_criterion

@register_criterion('multi_label_cross_entropy')
class MultiLabelCrossEntropyCriterion(PlatformNlpCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.task = task

    def get_loss(self):
        """Construct a criterion from command-line args."""
        output_layer = self.task.model.get_output()

        hidden_size = output_layer.shape[-1].value
        l2_loss = tf.constant(0.0)

        output_weights_layer = tf.get_variable(
            "output_weights_layer", [self.args.num_labels, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer())

        output_bias_layer = tf.get_variable(
            "output_bias_layer", [self.args.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            is_training = self.task.mode == tf.estimator.ModeKeys.TRAIN
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob= 1 - self.args.drop_prob)

            logits = tf.matmul(output_layer, output_weights_layer, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias_layer)
            l2_loss += tf.nn.l2_loss(output_weights_layer)
            l2_loss += tf.nn.l2_loss(output_bias_layer)

            probabilities = tf.nn.sigmoid(logits)
            multi_labels = tf.cast(self.task.labels, dtype=tf.float32)
            per_example_loss_multi_logits = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=multi_labels, logits=logits), axis=-1)
            per_example_loss = per_example_loss_multi_logits
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

