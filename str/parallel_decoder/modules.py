# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   created date: 2020/11/10
#   description:
#
#================================================================

import tensorflow as tf
from ..common.layers.transformer_layers import DecoderLayer
from ..common.layers.positional_encoding import positional_encoding_1d


class PDDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 rate=0.1,
                 embedded_inputs=False):
        super(PDDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedded_inputs = embedded_inputs

        if not embedded_inputs:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size,
                                                       d_model)
        self.pos_encoding = positional_encoding_1d(maximum_position_encoding,
                                                   d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_outputs, training):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        if not self.embedded_inputs:
            x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            layer_name = "pd_layer_{}".format(i)
            x, block1, block2 = self.dec_layers[i](x,
                                                   encoder_outputs,
                                                   training,
                                                   look_ahead_mask=None,
                                                   padding_mask=None,
                                                   cache=None)

            attention_weights['pd_decoder_layer{}_block1'.format(i +
                                                                 1)] = block1
            attention_weights['pd_decoder_layer{}_block2'.format(i +
                                                                 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
