# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for calculating loss, accuracy, and other model metrics.
Metrics:
 - Padded loss, accuracy, and negative log perplexity. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py
 - BLEU approximation. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
 - ROUGE score. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""
import functools
import tensorflow as tf


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size, mask_padding=True):
    """Calculate cross entropy loss while ignoring padding.
  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
    with tf.name_scope("loss"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
            soft_targets = tf.one_hot(tf.cast(labels, tf.int32),
                                      depth=vocab_size,
                                      on_value=confidence,
                                      off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(confidence * tf.math.log(confidence) + tf.cast(
                vocab_size - 1, tf.float32) * low_confidence * tf.math.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

        if mask_padding:
            weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        else:
            weights = tf.ones_like(labels, dtype=tf.float32)
        return xentropy * weights, weights


def transformer_loss(labels, logits, smoothing, vocab_size, mask_padding=True):
    """Calculates total loss containing cross entropy with padding ignored.
  Args:
    labels: Tensor of size [batch_size, length_labels]
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
    mask_padding: Boolean. Wether to mask padded labels. Default True.
  Returns:
    A scalar float tensor for loss.
  """
    xentropy, weights = padded_cross_entropy_loss(logits, labels, smoothing, vocab_size, mask_padding)
    return tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
