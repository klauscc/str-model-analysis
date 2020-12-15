# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/08
#   description:
#
#================================================================

import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(length, d_model):
    """
        Args:
            length (int): length of positions.
            d_model (int): channel of the model.

        Returns: Tensor of shape (1, length, d_model). positional encoding.
    """
    angle_rads = get_angles(
        np.arange(length)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def positional_encoding_2d(height, width, d_model):
    """positional encoding for 2D feature map (h, w, d_model).
    reference: https://arxiv.org/pdf/1908.11415.pdf
    github: https://github.com/wzlxjtu/PositionalEncoding2D

    Args:
        height (int): height of the positions.
        width (int): width of the positions.
        d_model (int): dimension of the model.

    Returns: Tensor of shape (height, width, d_model). 2d positional encoding.

    """
    half_d = d_model // 2
    angle_x = get_angles(
        np.arange(width)[np.newaxis, :, np.newaxis],
        np.arange(d_model // 4)[np.newaxis, np.newaxis, :], half_d)    # (1, width, d_model/4)
    angle_y = get_angles(
        np.arange(height)[:, np.newaxis, np.newaxis],
        np.arange(d_model // 4)[np.newaxis, np.newaxis, :], half_d)    # (height, 1, d_model/4)

    pos_encoding = np.zeros([height, width, d_model])
    pos_encoding[:, :, 0:half_d:2] = np.sin(angle_x)
    pos_encoding[:, :, 1:half_d:2] = np.cos(angle_x)
    pos_encoding[:, :, half_d::2] = np.sin(angle_y)
    pos_encoding[:, :, half_d + 1::2] = np.cos(angle_y)
    return tf.cast(pos_encoding, dtype=tf.float32)
