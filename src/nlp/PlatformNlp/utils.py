import importlib.util
import logging
import os
import sys
import warnings
import tensorflow as tf
from typing import Callable, Dict, List, Optional
import sklearn.metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp


logger = logging.getLogger(__name__)


MANIFOLD_PATH_SEP = "|"


def split_paths(paths: str) -> List[str]:
    return paths.split(os.pathsep) if "://" not in paths else paths.split(MANIFOLD_PATH_SEP)


def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(
                os.path.dirname(__file__), "..", args.user_dir
            )
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def total_sample(file_name):
    sample_nums = 0
    for _ in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums

# 读取tfrecords文件
def decode_from_tfrecords(filename, name_to_features, seq_length=128, num_epoch=None):
    if not isinstance(filename, list):
        filename = [filename]
    filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epoch)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features=name_to_features)
    # tf.Example only supports tf.int64, but the CPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    if 'input_ids' in example:
        input_ids = example['input_ids']
    else:
        input_ids = None
    if isinstance(input_ids, tf.SparseTensor):
        input_ids = tf.sparse_tensor_to_dense(input_ids)
    label = example['label_ids']
    return input_ids, label


def get_real_label(label, num):
    labels = []
    if label is None:
        return labels
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num):
            label_i = sess.run(label)
            labels.append(label_i)
    return labels

# 得到AUC的值 #
def Auc_value(y_pred, y_real, classes):
    classes = [int(c) for c in classes]
    if len(classes) > 2:
        y_real_label = label_binarize(y_real, classes=classes)
        y_pred_label = label_binarize(y_pred, classes=classes)
    else:
        y_real_label = np.zeros((len(y_real), 2), dtype=np.float32)
        y_pred_label = np.zeros((len(y_pred), 2), dtype=np.float32)
        for i in range(len(y_real)):
            if (y_real[i] == 0):
                y_real_label[i] = np.array([1.0, 0.0])
            else:
                y_real_label[i] = np.array([0.0, 1.0])

        for i in range(len(y_pred)):
            if (y_pred[i] == 0):
                y_pred_label[i] = np.array([1.0, 0.0])
            else:
                y_pred_label[i] = np.array([0.0, 1.0])

    y_pred = np.array(y_pred_label)
    n_classes = len(classes)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_real_label[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real_label.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc["macro"], roc_auc["micro"]


def calculate_label(y_pred, y_real, classes):
    assert len(y_pred) == len(y_real), "pred num and real num should be equal"
    precision = sklearn.metrics.precision_score(y_true=y_real, y_pred=y_pred, average='macro')
    recall = sklearn.metrics.recall_score(y_true=y_real, y_pred=y_pred, average='macro')
    accuracy = sklearn.metrics.accuracy_score(y_true=y_real, y_pred=y_pred)
    f1_score = sklearn.metrics.f1_score(y_true=y_real, y_pred=y_pred, average='macro')
    classify_report = sklearn.metrics.classification_report(y_true=y_real, y_pred=y_pred)
    auc_macro, auc_micro = Auc_value(y_pred, y_real, classes)
    return precision, recall, accuracy, f1_score, auc_micro, auc_macro, classify_report


def calculate_multi_label(y_pred, y_real):
    assert len(y_pred) == len(y_real), "pred num and real num should be equal"
    all_num = len(y_pred)
    correct_num = 0
    less_num = 0
    for i in range(len(y_pred)):
        real = y_real[i]
        pred = y_pred[i]

        same = True
        for j in range(len(real)):
            if real[j] != pred[j]:
                same = False
                break
        if same:
            correct_num += 1

        less = True
        for j in range(len(real)):
            if real[j] != pred[j]:
                if real[j] == 0 and pred[j] == 1:
                   less = False
                   break
        if less:
            less_num += 1

    exact_accu = correct_num / all_num
    less_accu = less_num / all_num

    return exact_accu, less_accu

