# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_data.py
# @Author: zys
# @Time  : 2022/11/23 14:15

import os
import csv
import json
import tensorflow as tf
import tokenization

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def _read_json(self, input_file, quotechar=None):
      with tf.gfile.Open(input_file, "r") as f:
          lines = []
          for line in f.readlines():
              if line == "" or line is None:
                  continue
              lines.append(json.loads(line.strip()))
          return lines

class NerProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")))

    def get_labels(self):
        clue_labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        res = ['O'] + [p + '-' + l for p in ['B', 'M', 'E', 'S'] for l in clue_labels]
        # 此处也可改为从label.csv文件中读取，方便后面做模型推理，这里为方便理解，做如上修改。
        return res

    def _create_examples(self, lines):
        """See base class."""
        # Finally examples is a list of InputSample where
        # InputSample.guid = "0", "1"...;
        # InputSample.text_a = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'
        # InputSample.text_b = None
        # InputSample.label = ['B-company', 'M-company', 'M-company', 'E-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'M-name', 'E-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s" % (i) if 'id' not in line else line['id']
            text_a = tokenization.convert_to_unicode(line['text'])
            label = ['O'] * len(text_a)
            if 'label' in line:
                for l, words in line['label'].items():
                    for word, indices in words.items():
                        for index in indices:
                            if index[0] == index[1]:
                                label[index[0]] = 'S-' + l
                            else:
                                label[index[0]] = 'B-' + l
                                label[index[1]] = 'E-' + l
                                for i in range(index[0] + 1, index[1]):
                                    label[i] = 'M-' + l
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def _create_examples_train(self, lines):
        """See base class."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s" % (i) if 'id' not in line else line['id']
            text_a = tokenization.convert_to_unicode(line['text'])
            label = ['O'] * len(text_a)
            if 'label' in line:
                for l, words in line['label'].items():
                    for word, indices in words.items():
                        for index in indices:
                            if index[0] == index[1]:
                                label[index[0]] = 'S-' + l
                            else:
                                label[index[0]] = 'B-' + l
                                label[index[1]] = 'E-' + l
                                for i in range(index[0] + 1, index[1]):
                                    label[i] = 'M-' + l
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

if __name__ == '__main__':
    processor = NerProcessor()
    data_dir = "ner_dataset/ner"
    train_examples = processor.get_train_examples(data_dir)
    dev_examples = processor.get_dev_examples(data_dir)
    test_examples = processor.get_test_examples(data_dir)
    print()