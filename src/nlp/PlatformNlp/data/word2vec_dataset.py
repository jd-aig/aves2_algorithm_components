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
    """A single training/test example for word2vec model."""

    def __init__(self, guid, tokens):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the sequence.
        """
        self.guid = guid
        self.tokens = tokens


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
                 is_real_example=True):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


@register_dataset("word2vec")
class Word2vecDataset(BaseDataset):
    """Loader for MultiClass dDataset"""

    def __init__(self, args):
        self.args = args
        self.max_seq_length = 200 if not args.max_seq_length or args.max_seq_length <= 0 else args.max_seq_length

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
                line = tokenization.convert_to_unicode(line)
                tokens = tokenizer.tokenize(line)
                if set_type == "train":
                    if len(tokens) < (2 * args.skip_window + 1):
                        continue
                    if len(tokens) > args.max_seq_length:
                        tokens = tokens[:args.max_seq_length]
                    if len(tokens) <= (2 * args.skip_window + 1):
                        continue
                lines.append(tokens)
            shuffle_index = list(range(len(lines)))
            random.shuffle(shuffle_index)
            for i in range(len(lines)):
                shuffle_i = shuffle_index[i]
                line_i = lines[shuffle_i]
                all_lines.append(line_i)
            del lines

        examples = []
        for (i, line) in enumerate(all_lines):
            # Only the test set has a header
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, tokens=line))

        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example, args, tokenizer)

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
            features["label_ids"] = create_int_feature(feature.label_ids)
            features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def builder(self, tfrecord_file, is_training, batch_size, drop_remainder, args):
        name_to_features = {
            "input_ids": tf.VarLenFeature(dtype=tf.int64),
            "label_ids": tf.VarLenFeature(dtype=tf.int64),
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

    def convert_single_example(self, ex_index, example, args, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        tokens = []

        tokens_a = example.tokens
        tokens.extend(tokens_a)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if self.args.type != "train":
            ids = input_ids
            labels = input_ids
            feature = InputFeatures(
                input_ids=ids,
                label_ids=labels,
                is_real_example=True)
        return feature

        num_skips = args.num_skips
        skip_window = args.skip_window
        while skip_window > 1 and len(input_ids) <= (2 * skip_window + 1):
            skip_window = int(skip_window / 2)

        if skip_window <= 1:
            return None

        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        data_index = 0
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(input_ids[data_index])
            data_index = (data_index + 1) % len(input_ids)

        ids = []
        labels = []
        for i in range(len(input_ids) // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                ids.append(buffer[skip_window])
                labels.append(buffer[target])
            buffer.append(input_ids[data_index])
            data_index = (data_index + 1) % len(input_ids)

        feature = InputFeatures(
            input_ids=ids,
            label_ids=labels,
            is_real_example=True)
        return feature
