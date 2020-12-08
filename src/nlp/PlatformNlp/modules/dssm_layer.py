import tensorflow as tf
from PlatformNlp.modules.utils import get_shape_list, create_initializer
from PlatformNlp.modules.batch_norm import batch_normalization
from PlatformNlp.modules.drop_out import dropout
from PlatformNlp.modules.cosine_score import get_cosine_score


def dssm_layer(query_ids, doc_ids, hidden_sizes, act_fn, is_training, max_seq_length, embedding_size, initializer_range, dropout_prob):
    shape = get_shape_list(query_ids, expected_rank=[2, 3])
    if len(shape) == 3:
        query_ids = tf.reshape(query_ids, [-1, shape[1] * shape[2]])
        doc_ids = tf.reshape(doc_ids, [-1, shape[1] * shape[2]])

    for i in range(0, len(hidden_sizes) - 1):
        query_ids = tf.layers.dense(query_ids, hidden_sizes[i], activation=act_fn,
                                    name="query_{}".format(str(i)),
                                    kernel_initializer=create_initializer(initializer_range))
        doc_ids = tf.layers.dense(doc_ids, hidden_sizes[i], activation=act_fn,
                                    name="doc_{}".format(str(i)),
                                    kernel_initializer=create_initializer(initializer_range))

        if is_training:
            query_ids = dropout(query_ids, dropout_prob)
            doc_ids = dropout(doc_ids, dropout_prob)

    query_pred = act_fn(query_ids)
    doc_pred = act_fn(doc_ids)
    cos_sim = get_cosine_score(query_pred, doc_pred)
    cos_sim_prob = tf.clip_by_value(cos_sim, 1e-8, 1.0)
    prob = tf.concat([query_pred, doc_pred], axis=1)
    return query_pred, doc_pred, prob





