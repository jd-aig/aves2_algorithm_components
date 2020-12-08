import math
import tensorflow as tf

from PlatformNlp.criterions.platform_criterion import PlatformNlpCriterion
from PlatformNlp.criterions import register_criterion
from tensorflow.contrib import crf


@register_criterion('ner')
class NerLossCriterion(PlatformNlpCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.task = task

    def get_loss(self):
        """Construct a criterion from command-line args."""
        sequence_output = self.task.model.get_output()
        input_ids = self.task.features["input_ids"]
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
        sequence_output = tf.reshape(sequence_output, shape=[-1, self.args.hidden_size])

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("logits_w", shape=[self.args.hidden_size, self.args.num_classes],
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("logits_b", shape=[self.args.num_classes], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            pred = tf.nn.xw_plus_b(sequence_output, w, b)
            pred = tf.reshape(pred, [-1, self.args.max_seq_length, self.args.num_classes])

        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            trans = tf.get_variable(
                "transitions",
                shape=[self.args.num_classes, self.args.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            if self.task.labels is None:
                loss = None
                log_likelihood = None
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=pred,
                    tag_indices=self.task.labels,
                    transition_params=trans,
                    sequence_lengths=lengths)
                loss = tf.reduce_mean(-log_likelihood)

        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=pred, transition_params=trans, sequence_length=lengths)

        return (loss, None, pred, pred_ids)

