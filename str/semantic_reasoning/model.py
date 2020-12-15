# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/09
#   description:
#
#================================================================

import re
import numpy as np
import tensorflow as tf
keras = tf.keras

from ..common.layers import transformer_layers, positional_encoding
from ..common import network


class SemanticReasonNet(keras.Model):
    """docstring for SemanticReasonNet"""

    def __init__(self, params):
        super(SemanticReasonNet, self).__init__()
        self.params = params

        input_shape = list(params.tar_img_size) + [3]
        self.backbone = network.build_backbone(input_shape, params.d_model, params.bn_momentum)
        self.self_attention_layers = [
            transformer_layers.EncoderLayer(params.d_model, params.num_heads, params.dff,
                                            params.dropout_rate)
            for i in range(params.num_encoder_sa)
        ]
        self.parallel_visual_attention_module = ParallelVisualAttentionModule(
            params.max_n_chars, params.d_model, params.dict_size)

        if params.with_semantic_reasoning:
            self.semantic_reasoning = SemanticReasoningModule(params.max_n_chars, params.d_model,
                                                              params.dict_size, params.num_heads,
                                                              params.dff, params.dropout_rate,
                                                              params.num_decoder_sa)
            self.visual_semantic_fusion = VisualSemanticFusionModule(params.dict_size)

    def augment_data(self, data):
        # modify the data to make y and y_pred have the same shape.
        x, y = data
        idx = y
        y = (idx, idx, idx) if self.params.with_semantic_reasoning else idx
        data = (x, y)
        return data

    def train_step(self, data):
        data = self.augment_data(data)
        return super().train_step(data)

    def test_step(self, data):
        data = self.augment_data(data)
        return super().test_step(data)

    def call(self, inp, training):
        encoder_output = self.backbone(inp, training)
        h, w, c = encoder_output.shape[1:].as_list()
        pe = positional_encoding.positional_encoding_2d(h, w, c)
        encoder_output = encoder_output + pe

        # flatten hxw to L
        bs, h, w, d_model = encoder_output.shape.as_list()
        encoder_output = tf.reshape(encoder_output, (-1, h * w, d_model))    # (bs, L, d_model)

        for layer in self.self_attention_layers:
            encoder_output = layer(encoder_output, training=training, mask=None)

        glimpses, glimpse_logits = self.parallel_visual_attention_module(encoder_output,
                                                                         training=training)
        if self.params.with_semantic_reasoning:
            semantics, semantic_logits = self.semantic_reasoning(glimpse_logits, training=training)
            fused_logits = self.visual_semantic_fusion([glimpses, semantics])
            return glimpse_logits, semantic_logits, fused_logits
        else:
            return glimpse_logits


class ParallelVisualAttentionModule(keras.layers.Layer):
    """parallel visual attention module"""

    def __init__(self, max_n_chars, d_model, dict_size, with_classifier=True, *args, **kwargs):
        self.max_n_chars = max_n_chars
        self.d_model = d_model
        super(ParallelVisualAttentionModule, self).__init__(*args, **kwargs)

        self.char_orders = tf.cast(np.arange(max_n_chars), tf.int32)
        self.embedding = keras.layers.Embedding(max_n_chars, d_model, input_length=max_n_chars)

        self.Wo = keras.layers.Dense(d_model, name='Wo')
        self.Wv = keras.layers.Dense(d_model, name='Wv')
        self.We = keras.layers.Dense(1, name='We')

        self.softmax = keras.layers.Softmax(axis=(2))    #input_shape: (bs, T, L). L = h x w

        self.with_classifier = with_classifier
        if self.with_classifier:
            self.classifier = keras.layers.Dense(dict_size, name='glimpse_classifier')

    def call(self, inp, training):
        """
            Args:
                inp: Tensor of shape (bs, L, d_model).

            Returns: Tensor of shape (bs, max_n_chars, d_model). visual feature `glimpses`.
                
        """

        O = self.embedding(self.char_orders)    # (T,d_model)
        w_o = self.Wo(O)    # (T, d_model)
        w_o = w_o[tf.newaxis, :, tf.newaxis, :]    # (1, T, 1, d_model)

        w_v = self.Wv(inp)    # (bs, L, d_model)
        w_v = w_v[:, tf.newaxis, :, :]    # (bs, 1, L, d_model)

        e = self.We(tf.nn.tanh(w_o + w_v))    # (bs, T, L, 1)
        e = tf.squeeze(e, axis=-1)    # (bs,T, L)
        alpha = self.softmax(e)
        visual_features = tf.matmul(alpha,
                                    inp)    # (bs, T, L) x (bs, L, d_model) = (bs, T, d_model)
        if self.with_classifier:
            logits = self.classifier(visual_features)    # (bs, T, dict_size)
            return visual_features, logits
        else:
            return visual_features


class SemanticReasoningModule(keras.layers.Layer):
    """docstring for SemanticReasoningModule"""

    def __init__(self, max_n_chars, d_model, dict_size, num_heads, dff, dropout_rate,
                 num_self_attention, *args, **kwargs):
        super(SemanticReasoningModule, self).__init__(*args, **kwargs)

        self.char_embedding = keras.layers.Embedding(dict_size, d_model, input_length=max_n_chars)

        self.self_attention_layers = [
            transformer_layers.EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for i in range(num_self_attention)
        ]
        self.classifier = keras.layers.Dense(dict_size)

    def call(self, glimpse_logits, training):
        """TODO: Docstring for call.

        Args:
            glimpse_logits (Tensor of shape (bs, T, dict_size) ): logits from glimpses (visual features).

        Returns: TODO

        """
        pred_char_ids = tf.argmax(glimpse_logits, axis=2)
        semantics = self.char_embedding(pred_char_ids)
        for layer in self.self_attention_layers:
            semantics = layer(semantics, training=training, mask=None)
        logits = self.classifier(semantics)
        return semantics, logits


class VisualSemanticFusionModule(keras.layers.Layer):
    """fuse visual feature and semantic feature to make the final prediction"""

    def __init__(self, dict_size):
        super(VisualSemanticFusionModule, self).__init__()

        self.Wz = keras.layers.Dense(1)
        self.classifier = keras.layers.Dense(dict_size)

    def call(self, features):
        glimpse, semantics = features    # each feature: (bs, max_n_chars, d_model)
        feature = tf.concat([glimpse, semantics], axis=-1)    # (bs, max_n_chars, 2*d_model)
        z = tf.sigmoid(self.Wz(feature))    # (bs, max_n_chars, 1)
        fused_feature = z * glimpse + (1 - z) * semantics
        logits = self.classifier(fused_feature)
        return logits    # (bs, max_n_chars, dict_size)
