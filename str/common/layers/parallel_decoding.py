# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/08/12
#   description:
#
#================================================================

import tensorflow as tf

from ..networks.unet import unet


class CAMVisualFeatures(tf.keras.layers.Layer):
    """CAM: [Decoupled Attention Network for Text Recognition](https://arxiv.org/abs/1912.10205)
      
        Unet to predict the attention map for each characters.
      Gather the visual features with the attention maps.
    """

    def __init__(self, max_n_chars, input_shape, *args, **kwargs):
        super(CAMVisualFeatures, self).__init__(*args, **kwargs)
        self.max_n_chars = max_n_chars

        self.inp_shape = input_shape
        self.cam_net = unet(3, max_n_chars, input_shape=input_shape)

    def call(self, encoder_outputs, training):
        # encoder_outputs: bs x h x w x c
        h, w, _ = self.inp_shape
        c = encoder_outputs.shape[-1]
        encoder_outputs = tf.reshape(encoder_outputs, [-1, h, w, c])
        att_maps = self.cam_net(encoder_outputs, training)    # bs x h x w x max_n_chars
        att_maps = tf.reshape(att_maps, [-1, h * w, self.max_n_chars])
        att_maps = tf.math.softmax(att_maps, axis=1)
        att_maps = tf.reshape(att_maps, [-1, h, w, self.max_n_chars])
        gathered_features = encoder_outputs[:, :, :, tf.newaxis, :] * \
                att_maps[:, :, :, :, tf.newaxis] # (bs, h, w, 1, c) x (bs, h, w, max_n_chars, 1) -> (bs, h, w, max_n_chars, c)
        visual_features = tf.reduce_sum(gathered_features, axis=(1, 2))    # (bs, max_n_chars, c)
        return visual_features
