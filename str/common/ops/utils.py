# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/08
#   description:
#
#================================================================

import tensorflow as tf


def resize_keep_aspect_ratio(image, tar_w, max_h):
    src_h = tf.shape(image)[0]
    src_w = tf.shape(image)[1]
    tar_h = tf.cast(src_h * tar_w / src_w, tf.int32)
    if tar_h > max_h:
        tar_h = max_h
    if tar_h < 5: # avoid tar_h = 0
        tar_h = 5
    return tf.image.resize(image, (tar_h, tar_w))


def random_choice_1d(array, probs=None):
    random_num = tf.random.uniform(shape=[])
    if probs is None:
        probs = [1. / len(array)] * len(array)
    accumulated_prob = 0
    choosen = array[-1]
    picked = False
    for i, prob in enumerate(probs):
        accumulated_prob += prob
        if random_num < accumulated_prob:
            if not picked:
                choosen = array[i]
                picked = True
    return choosen
