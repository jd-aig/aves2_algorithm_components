import tensorflow as tf


def get_cosine_score(query_arr, doc_arr):
    # query_norm = sqrt(sum(each x^2))
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
    cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
    return cos_scores