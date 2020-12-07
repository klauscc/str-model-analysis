# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/30
#   description:
#
#================================================================
import tensorflow as tf
import tensorflow.keras as keras

from ..networks.unet import unet
from ..samplers import bilinear_sampler


class PixelFlow(keras.layers.Layer):
    """docstring for PixelFlow"""

    def __init__(self, input_shape):
        super(PixelFlow, self).__init__()

        self.flow_estimator = unet(
            n_scales=5,
            out_nc=3,
            input_shape=input_shape,
            last_initializer=[tf.keras.initializers.Zeros(),
                              tf.keras.initializers.Zeros()])

    def call(self, img, training):
        flows = self.flow_estimator(img, training)
        x_flow = flows[..., 0]
        y_flow = flows[..., 1]
        visual_map = flows[..., 2]

        h = tf.shape(img)[1]
        w = tf.shape(img)[2]

        # apply pixel flow
        grid_x, grid_y = tf.meshgrid(tf.range(w), tf.range(h))
        x = tf.cast(grid_x, tf.float32) + x_flow
        y = tf.cast(grid_y, tf.float32) + y_flow
        x = 2 * (x / tf.cast(w, tf.float32) - 0.5)
        y = 2 * (y / tf.cast(h, tf.float32) - 0.5)    # [bs, h, w]
        grid = tf.stack([x, y], axis=3)    # [bs, h, w, 2]

        flow_img = bilinear_sampler(img, grid)

        # apply visualization map
        visual_map = tf.tanh(visual_map) + 1    # scale visual_map to [0, 2].
        flow_img = flow_img * visual_map[:, :, :, tf.newaxis]
        return flow_img
