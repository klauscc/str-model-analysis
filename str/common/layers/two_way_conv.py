# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   created date: 2020/10/29
#   description:
#
#================================================================

import tensorflow as tf
keras = tf.keras


def rot90(imgs):
    """rotate 90 degree anti-clockwise

    Args:
        imgs (Tensor): imgs of shape (bs, h, w, c) 

    Returns:  Tensor. Rotated imgs of shape (bs, w, h, c) 

    """
    # transpose -> horizontal reflection.
    imgs = tf.transpose(imgs, perm=(0, 2, 1, 3))
    imgs = tf.reverse(imgs, axis=[1])
    return imgs


def rot270(imgs):
    """rotate 90 degree anti-clockwise

    Args:
        imgs (Tensor): imgs of shape (bs, h, w, c) 

    Returns:  Tensor. Rotated imgs of shape (bs, w, h, c) 

    """
    # transpose -> vertical reflection.
    imgs = tf.transpose(imgs, perm=(0, 2, 1, 3))
    imgs = tf.reverse(imgs, axis=[2])
    return imgs


class TwoWayConv(keras.layers.Layer):
    """docstring for TwoWayConv"""
    def __init__(self, conv, merge_mode='max', **kwargs):
        super(TwoWayConv, self).__init__(**kwargs)

        self.conv = conv
        self.merge_mode = merge_mode

    def call(self, x):
        # x: (bs, h, w, c)
        y1 = self.conv(x)

        # rot 270 degrees anti-clockwise. (rot90 clockwise)
        x_rot = rot270(x)

        y2 = self.conv(x_rot)  # (bs, w, h, c)

        # rot 90 degrees anti-clockwise
        y2 = rot90(y2)

        if self.merge_mode == 'max':
            y = tf.maximum(y1, y2)
        elif self.merge_mode == 'mean':
            y = (y1 + y2) / 2
        return y
