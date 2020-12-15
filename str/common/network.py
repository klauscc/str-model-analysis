# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/14
#   description:
#
#================================================================

import tensorflow as tf
keras = tf.keras

from ..common.layers import transformer_layers, positional_encoding
from ..common.layers.two_way_conv import TwoWayConv


class SelfAttentionEncoder(keras.layers.Layer):
    """perform self attention on the output of backbone"""
    def __init__(self,
                 d_model,
                 num_heads,
                 num_sa,
                 dff,
                 dropout_rate,
                 *args,
                 attend_1d=False,
                 keep_shape=False,
                 **kwargs):
        super(SelfAttentionEncoder, self).__init__(*args, **kwargs)
        self.self_attention_layers = [
            transformer_layers.EncoderLayer(d_model, num_heads, dff,
                                            dropout_rate)
            for i in range(num_sa)
        ]
        self.keep_shape = keep_shape
        self.attend_1d = attend_1d
        self.d_model = d_model

        if attend_1d:
            self.h_proj = tf.keras.layers.Dense(d_model)

    def call(self, backbone_output, training, mask=None):
        h, w, c = backbone_output.shape[1:].as_list()

        if self.attend_1d:
            encoder_output = tf.transpose(backbone_output,
                                          (0, 2, 3, 1))  # (bs, w,c,h)
            encoder_output = tf.reshape(encoder_output, (-1, w, h * c))
            encoder_output = self.h_proj(encoder_output)  # (bs, w, c)
            pe = positional_encoding.positional_encoding_1d(w, self.d_model)
            encoder_output = encoder_output + pe
        else:  # 2d attention
            pe = positional_encoding.positional_encoding_2d(h, w, c)
            encoder_output = backbone_output + pe
            # flatten hxw to L
            _, h, w, d_model = encoder_output.shape.as_list()
            encoder_output = tf.reshape(
                encoder_output, (-1, h * w, d_model))  # (bs, L, d_model)

        # self attention
        for layer in self.self_attention_layers:
            encoder_output = layer(encoder_output,
                                   training=training,
                                   mask=mask)

        if not self.attend_1d and self.keep_shape:
            encoder_output = tf.reshape(encoder_output, (-1, h, w, d_model))
        return encoder_output


def build_backbone(input_shape,
                   d_model,
                   bn_momentum,
                   two_way_conv=False,
                   merge_mode='max'):
    """
        Args:
            input_shape (list). The input shape. e.g. (64,256,3)
            d_model (int). The dimension of model.
            bn_momentum (float). The momentum for bn layers.
            two_way_conv (Boolean). Whether two add two way convolution.

        Returns:
            keras.Model. The model output is 1/8 of the input size. e.g. (8,32,d_model)
    """
    resnet = keras.applications.ResNet50(include_top=False,
                                         input_shape=input_shape)
    inp = resnet.input
    c3 = resnet.get_layer('conv3_block4_out').output
    c4 = resnet.get_layer('conv4_block6_out').output
    c5 = resnet.get_layer('conv5_block3_out').output

    p5 = keras.layers.Conv2D(d_model, 1, padding='same', name='P5')(c5)

    p5_up = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(p5)
    p4 = keras.layers.Conv2D(d_model, 1, padding='same')(c4)
    p4 = keras.layers.Add(name='P4')([p4, p5_up])

    p4_up = keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(p4)
    p3 = keras.layers.Conv2D(d_model, 1, padding='same')(c3)
    p3 = keras.layers.Add(name='P3')([p3, p4_up])

    if two_way_conv:
        p3 = TwoWayConv(keras.layers.Conv2D(d_model, 3, padding='same'),
                        merge_mode=merge_mode)(p3)

    backbone = keras.Model(inp, p3)

    #reset bn moving_mean and moving_variance
    for i, layer in enumerate(backbone.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = bn_momentum
            weights = layer.get_weights()
            new_weights = [
                weights[0],  # gamma
                # tf.ones_like(weights[0]),
                weights[1],  # beta
                # tf.zeros_like(weights[1]),
                tf.zeros_like(weights[2]),  # moving_mean
                tf.ones_like(weights[3])  # moving_variance
            ]
            layer.set_weights(new_weights)

    return backbone
