import math
import tensorflow as tf


from PlatformNlp.criterions.platform_criterion import PlatformNlpCriterion
from PlatformNlp.criterions import register_criterion


@register_criterion('nce')
class NceLossCriterion(PlatformNlpCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.task = task

    def get_loss(self):
        """Construct a criterion from command-line args."""
        embedding_output = self.task.model.get_output()
        embedding_output = tf.reshape(embedding_output, [-1, self.args.embedding_size])
        embedding_table = self.task.model.get_embedding()
        labels = tf.sparse_tensor_to_dense(self.task.labels)
        labels = tf.reshape(labels, [-1])
        labels = tf.expand_dims(labels, axis=[-1])

        with tf.variable_scope("nce", reuse=tf.AUTO_REUSE):
            self.nce_weight = tf.get_variable("nce_weight", initializer=tf.truncated_normal(
                [self.args.vocab_size, self.args.embedding_size], stddev=1.0 / math.sqrt(self.args.embedding_size)))

            self.nce_biases = tf.get_variable("nce_biases", initializer=tf.zeros([self.args.vocab_size]))
            per_example_loss = tf.nn.nce_loss(weights=self.nce_weight,
                                              biases=self.nce_biases,
                                              labels=labels,
                                              inputs=embedding_output,
                                              num_sampled=self.args.num_sampled,
                                              num_classes=self.args.vocab_size)

            loss = tf.reduce_mean(per_example_loss)
            logits = tf.matmul(embedding_output, self.nce_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, self.nce_biases)

            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(embedding_table), 1, keep_dims=True)
            )
            normed_embedding = embedding_table / vec_l2_model
            input_ids = self.task.features["input_ids"]
            input_ids = tf.sparse_tensor_to_dense(input_ids)
            normed_embedding_output = tf.nn.embedding_lookup(normed_embedding, input_ids)
            normed_embedding_output = tf.reshape(normed_embedding_output, [-1, self.args.embedding_size])
            similarity = tf.matmul(normed_embedding_output, normed_embedding, transpose_b=True)
            similarity = tf.reshape(similarity, [self.args.batch_size, -1, self.args.vocab_size])
            probabilities = similarity

        return (loss, per_example_loss, logits, probabilities)

