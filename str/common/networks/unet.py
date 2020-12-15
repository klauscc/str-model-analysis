# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/28
#   description:
#
#================================================================
import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters,
                               size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(filters,
                                        size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet(
    n_scales,
    out_nc,
    input_shape,
    init_filters=64,
    size=3,
    max_filters=512,
    last_initializer=None,
):
    down_filters = [min(init_filters * (2**i), max_filters) for i in range(n_scales)]
    down_stack = [
        downsample(flt, size, apply_batchnorm=False if i == 0 else True)
        for i, flt in enumerate(down_filters)
    ]
    up_filters = reversed(down_filters[:-1])
    up_stack = [
        upsample(flt, size, apply_dropout=True if i >= n_scales // 2 else False)
        for i, flt in enumerate(up_filters)
    ]
    if last_initializer is not None:
        kernel_initializer = last_initializer[0]
        bias_initializer = last_initializer[1]
    else:
        kernel_initializer = 'glorot_uniform'
        bias_initializer = 'zeros'

    last = tf.keras.layers.Conv2DTranspose(out_nc,
                                           size,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer)    # (bs, 256, 256, 3)

    inp = tf.keras.layers.Input(shape=input_shape)
    x = inp

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inp, outputs=x)
