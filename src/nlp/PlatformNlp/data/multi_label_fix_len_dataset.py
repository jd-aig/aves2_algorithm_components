# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import six
import codecs
import random
import collections
import tensorflow as tf
import PlatformNlp.tokenization as tokenization
from PlatformNlp.data.base_dataset import BaseDataset
from PlatformNlp.data import register_dataset


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 label_ids,
                 input_mask,
                 segment_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_real_example = is_real_example


@register_dataset("multi_label")
class MultiLabelFixLenDataset(BaseDataset):
    """Loader for MultiClass dDataset"""

    def __init__(self, args):
        self.args = args
        self.max_seq_length = 200 if not args.max_seq_length or args.max_seq_length <= 0 else args.max_seq_length
        self.num_labels = 2 if args.num_labels is None or args.num_labels <= 0 else args.num_labels
        if args.label_file is not None and os.path.exists(args.label_file):
            self.label_mapping = six.moves.cPickle.load(open(args.label_file, 'rb'))
        else:
            self.label_mapping = {}

    def build_dataset(self, args, tokenizer):
        set_type = args.type
        data_file = args.data_file
        label_file = args.label_file
        output_file = args.output_file
        if not os.path.exists(data_file):
            raise FileExistsError("{} does not exists!!!".format(data_file))
        if os.path.exists(output_file):
            os.remove(output_file)
        all_lines = []
        with codecs.open(data_file, "r", 'utf-8', errors='ignore') as f:
            lines = []
            for line in f:
                line = line.strip('\n')
                line = line.strip("\r")
                line = line.split(',')
                if len(line) < 2:
                    continue
                lines.append(line)
            shuffle_index = list(range(len(lines)))
            random.shuffle(shuffle_index)
            for i in range(len(lines)):
                shuffle_i = shuffle_index[i]
                if len(lines[i]) != 2:
                    continue
                line_i = [str(lines[shuffle_i][0]), str(lines[shuffle_i][1])]
                all_lines.append(line_i)
            del lines

        examples = []
        for (i, line) in enumerate(all_lines):
            # Only the test set has a header
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            if set_type.lower() == "train":
                labels = str(label).split("+")
                for l in labels:
                    if l not in self.label_mapping:
                        self.label_mapping[l] = len(self.label_mapping)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        if set_type.lower() != "train":
            if not os.path.exists(label_file):
                raise EnvironmentError("no labels exists !!!!!")
            self.label_mapping = six.moves.cPickle.load(open(label_file, 'rb'))
        else:
            with open(label_file, 'wb') as f:
                six.moves.cPickle.dump(self.label_mapping, f)

        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example, self.label_mapping, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            def create_str_feature(value):
                if isinstance(value, str):
                    value = bytes(value, encoding='utf-8')
                f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def builder(self, tfrecord_file, is_training, batch_size, drop_remainder, args):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([self.num_labels], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
        args.name_to_features = name_to_features

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            # batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(tfrecord_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn

    def convert_single_example(self, ex_index, example, label_mapping, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                input_ids=[0] * self.max_seq_length,
                input_mask=[0] * self.max_seq_length,
                segment_ids=[0] * self.max_seq_length,
                label_id=[0] * self.num_labels,
                is_real_example=False)

        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        labels = str(example.label).split("+")
        label_ids = [0] * self.num_labels

        for label in labels:
            label_i = int(label_mapping.get(label, 0))
            label_ids[label_i] = 1
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            is_real_example=True)
        return feature
