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
from .mask import create_padding_mask
from ..label_converter import PAD_ID, EOS_ID


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding_1d(maximum_position_encoding,
                                                   d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,
             x,
             encoder_outputs,
             training,
             look_ahead_mask,
             padding_mask,
            pd_features=None,
             cache=None,
             loop_i=None):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if loop_i is not None:  # when predict, the decode is sequential.
            x += self.pos_encoding[:, loop_i:loop_i + 1, :]
            look_ahead_mask = look_ahead_mask[:, :, loop_i:loop_i +
                                              1, :loop_i + 1]
            if pd_features is not None:
                x += pd_features[:, loop_i:loop_i + 1, :]
        else:
            x += self.pos_encoding[:, :seq_len, :]
            if pd_features is not None:
                x += pd_features[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            layer_name = "layer_{}".format(i)
            layer_cache = cache[layer_name] if cache is not None else None
            x, block1, block2 = self.dec_layers[i](x,
                                                   encoder_outputs,
                                                   training,
                                                   look_ahead_mask,
                                                   padding_mask,
                                                   cache=layer_cache)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class PDDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 rate=0.1,
                 embedding_layer=None):
        super(PDDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding

        if embedding_layer is None:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size,
                                                       d_model)
        else:
            self.embedding = embedding_layer
        # self.abs_pos_embedding = tf.keras.layers.Embedding(
        # target_vocab_size, d_model)
        self.pos_encoding = positional_encoding_1d(maximum_position_encoding,
                                                   d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_outputs, training):

        bs = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        abs_pos_encoding = tf.tile(tf.expand_dims(tf.range(
            self.maximum_position_encoding),
                                                  axis=0),
                                   multiples=[bs, 1])

        # look_ahead_mask = create_padding_mask(x)
        look_ahead_mask = None
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # abs_pos_emb = self.abs_pos_embedding(abs_pos_encoding[:, :seq_len])
        # x += abs_pos_emb
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            layer_name = "pd_layer_{}".format(i)
            x, block1, block2 = self.dec_layers[i](
                x,
                encoder_outputs,
                training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=None,
                cache=None)

            attention_weights['pd_decoder_layer{}_block1'.format(i +
                                                                 1)] = block1
            attention_weights['pd_decoder_layer{}_block2'.format(i +
                                                                 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class JitterSeqPredicton(tf.keras.layers.Layer):
    """perform random replace, insert, delete of predictions"""
    def __init__(self,
                 dict_size,
                 jitter_prob=0.3,
                 insert_prob=0.2,
                 replace_prob=0.2,
                 delete_prob=0.2):
        super(JitterSeqPredicton, self).__init__()

        self.dict_size = dict_size
        self.jitter_prob = jitter_prob
        self.insert_prob = insert_prob
        self.replace_prob = replace_prob
        self.delete_prob = delete_prob

    def call(self, xs):
        bs = tf.shape(xs)[0]
        inp_len = tf.shape(xs)[1]

        res = tf.zeros([0, inp_len], dtype=xs.dtype)
        i = tf.constant(0)

        c = lambda i, res: tf.less(i, bs)
        b = lambda i, res: (
            i + 1, tf.concat([res, self.jitter(xs[i])[tf.newaxis, :]], axis=0))
        i, res = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(c,
                          b,
                          loop_vars=[i, res],
                          shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([None, None])
                          ]))
        return res

    def jitter(self, x):
        # x: (max_n_chars,). One sample
        orig_x = x
        dtype = x.dtype
        inp_len = tf.shape(x)[-1]
        if tf.random.uniform([]) > self.jitter_prob:
            return x

        # replace
        eos_idx = tf.cast(tf.where(x == EOS_ID), tf.int32)
        if tf.shape(eos_idx)[0] == 0:
            eos_idx = inp_len // 2
        else:
            eos_idx = tf.maximum(eos_idx[0, 0], 1)
            eos_idx = tf.minimum(eos_idx, inp_len - 4)
        p = tf.random.uniform([])
        if p < 0.33:
            jitter_values = tf.random.uniform([inp_len],
                                              minval=EOS_ID + 1,
                                              maxval=self.dict_size,
                                              dtype=dtype)
            keep_probs = tf.concat(
                [
                    tf.random.uniform([eos_idx], minval=0, maxval=1),
                    tf.ones([inp_len - eos_idx
                             ]),  # do not jitter [EOS] and after
                ],
                axis=-1)  # [bs, len]
            x = tf.where(keep_probs > self.insert_prob, x, jitter_values)

        # insert
        elif p < 0.66:
            insert_n = tf.random.uniform([],
                                         minval=1,
                                         maxval=tf.cast(
                                             tf.cast(eos_idx + 1, tf.float32) *
                                             self.insert_prob, tf.int32) + 2,
                                         dtype=tf.int32)
            insert_idxs = tf.random.shuffle(
                tf.range(0, eos_idx, dtype=tf.int32))[:insert_n]
            insert_idxs = tf.sort(insert_idxs)
            insert_values = tf.random.uniform([insert_n],
                                              minval=EOS_ID + 1,
                                              maxval=self.dict_size,
                                              dtype=dtype)
            inserted_x = tf.concat([x[:insert_idxs[0]], [insert_values[0]]],
                                   axis=-1)
            for i in tf.range(1, insert_n):
                inserted_x = tf.concat([
                    inserted_x, x[insert_idxs[i - 1]:insert_idxs[i]],
                    [insert_values[i]]
                ],
                                       axis=-1)

            inserted_x = tf.concat(
                [inserted_x, x[insert_idxs[insert_n - 1]:eos_idx + 1]],
                axis=-1)

            x = inserted_x

        # delete
        else:
            delete_n = tf.random.uniform([],
                                         minval=1,
                                         maxval=tf.cast(
                                             tf.cast(eos_idx + 1, tf.float32) *
                                             self.delete_prob, tf.int32) + 2,
                                         dtype=tf.int32)
            delete_idxs = tf.random.shuffle(
                tf.range(0, eos_idx, dtype=tf.int32))[:delete_n]
            delete_idxs = tf.sort(delete_idxs)
            del_x = x[:delete_idxs[0]]
            for i in tf.range(1, delete_n):
                del_x = tf.concat(
                    [del_x, x[delete_idxs[i - 1] + 1:delete_idxs[i]]], axis=-1)
            del_x = tf.concat(
                [del_x, x[delete_idxs[delete_n - 1] + 1:eos_idx + 1]], axis=-1)

            x = del_x

        if inp_len >= tf.shape(x)[-1]:
            x = tf.pad(x, [[0, inp_len - tf.shape(x)[-1]]],
                       constant_values=PAD_ID)
        else:
            x = orig_x
        return x
