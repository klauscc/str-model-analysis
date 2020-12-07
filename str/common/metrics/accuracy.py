# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/10
#   description:
#
#================================================================

import tensorflow as tf
keras = tf.keras
import re


def normalize_text(text):
    text = tf.strings.regex_replace(text, re.escape('[UNK]'), '')
    text = tf.strings.regex_replace(text, '\W',
                                    '')  # only preserve [0-9a-zA-Z_]
    text = tf.strings.lower(text)
    return text


class PerSampleAccuracy(keras.metrics.Accuracy):
    """docstring for PerSampleAccuracy"""
    def __init__(self, tokenizer, *args, **kwargs):
        super(PerSampleAccuracy, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self._fn = tf.function(self.sparse_categorial_accuracy,
                               experimental_relax_shapes=True)

    def sparse_categorial_accuracy(self, y_true, y_pred):
        """TODO: Docstring for sparse_categorial_accuracy.

            Args:
                y_true (Tensor ): shape (bs, max_n_chars). 
                y_pred (Tensor): logits. shape (bs, max_n_chars, dict_size).

            Returns: accuracy along batch.

        """
        y_true = tf.cast(y_true, tf.int32)
        if len(y_pred.shape.as_list()) == 3:
            y_pred = tf.argmax(y_pred, axis=2)  #(bs, max_n_chars)
        y_pred = tf.cast(y_pred, tf.int32)
        y_true = normalize_text(self.tokenizer.decode(y_true))
        y_pred = normalize_text(self.tokenizer.decode(y_pred))
        res = tf.equal(y_true, y_pred)
        return res
