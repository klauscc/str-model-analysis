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
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, ReLU, Dense
from ..samplers import bilinear_sampler


class TPS_STN(tf.keras.layers.Layer):
    """Rectification Network of RARE, namely TPS based STN"""

    def __init__(self, num_fiducial, inp_size, tar_size):
        super(TPS_STN, self).__init__()
        self.num_fiducial = num_fiducial
        self.inp_size = inp_size
        self.tar_size = tar_size

        self.localization_network = LocalizationNetwork(num_fiducial)
        self.grid_generator = GridGenerator(num_fiducial, tar_size)

    def call(self, x, training):
        c_prime = self.localization_network(x, training)
        p_prime = self.grid_generator(c_prime)    # (bs, n, 2)
        grid = tf.reshape(p_prime, (-1, self.tar_size[0], self.tar_size[1], 2))    # (bs, h, w, 2)
        x = bilinear_sampler(x, grid)
        return x


class LocalizationNetwork(tf.keras.layers.Layer):
    """Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height)"""

    def __init__(self, num_fiducial):
        super(LocalizationNetwork, self).__init__()
        self.num_fiducial = num_fiducial

        self.convs = tf.keras.Sequential([
            Conv2D(64, 3, (1, 1), padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),    # / 2
            Conv2D(128, 3, (1, 1), padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),    # / 4
            Conv2D(256, 3, (1, 1), padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),    # / 8
            Conv2D(512, 3, (1, 1), padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            GlobalAveragePooling2D(),    # [bs, 512] 
        ])

        self.localization_fc1 = Dense(256, activation='relu')
        self.localization_fc2 = Dense(num_fiducial * 2,
                                      kernel_initializer=tf.keras.initializers.Zeros(),
                                      bias_initializer=self.localization_fc2_bias_initializer())

        # Init fc2 in LocalizationNetwork

    def call(self, x, training):
        features = self.convs(x, training)
        c_prime = self.localization_fc1(features)    # (bs, 256)
        c_prime = self.localization_fc2(c_prime)    # (bs, num_fiducial*2)
        return tf.reshape(c_prime, (-1, self.num_fiducial, 2))

    def localization_fc2_bias_initializer(self):
        F = self.num_fiducial

        class FC2BiasInitializer(tf.keras.initializers.Initializer):
            """docstring for FC2BiasInitializer"""

            def __init__(self):
                super(FC2BiasInitializer, self).__init__()

            def __call__(self, shape, dtype):
                ctrl_pts_x = tf.linspace(-1.0, 1.0, F // 2)
                ctrl_pts_y_top = tf.linspace(0.0, -1.0, F // 2)
                ctrl_pts_y_bottom = tf.linspace(1.0, 0.0, F // 2)
                ctrl_pts_top = tf.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
                ctrl_pts_bottom = tf.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
                initial_bias = tf.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
                initial_bias = tf.reshape(initial_bias, [-1])
                return tf.cast(initial_bias, dtype=dtype)

        return FC2BiasInitializer()


class GridGenerator(tf.keras.layers.Layer):
    """Grid Generator of RARE, which produces P_prime by multipling T with P"""

    def __init__(self, num_fiducial, tar_size):
        super(GridGenerator, self).__init__()
        self.num_fiducial = num_fiducial
        self.tar_size = tar_size
        self.eps = 1e-6

        self.C = self._build_C(self.num_fiducial).astype(np.float32)    # F x 2
        self.P = self._build_P(tar_size).astype(np.float32)

        self.inv_delta_C = self._build_inv_delta_C(num_fiducial, self.C).astype(np.float32)
        self.P_hat = self._build_P_hat(num_fiducial, self.C, self.P).astype(np.float32)

    def call(self, c_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = tf.shape(c_prime)[0]
        batch_inv_delta_C = tf.tile(self.inv_delta_C[np.newaxis, :, :],
                                    [batch_size, 1, 1])    # (bs, F+3, F+3)
        batch_P_hat = tf.tile(self.P_hat[np.newaxis, :, :], [batch_size, 1, 1])    # (bs, n, F+3)
        c_prime_with_zeros = tf.concat([c_prime, tf.zeros([batch_size, 3, 2])],
                                       axis=1)    # batch_size x F+3 x 2
        batch_T = tf.matmul(batch_inv_delta_C, c_prime_with_zeros)    # (bs, F+3, 2)
        batch_P_prime = tf.matmul(batch_P_hat, batch_T)    # (bs, n, 2)
        return batch_P_prime

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, F // 2)
        ctrl_pts_y_top = -1 * np.ones(F // 2)
        ctrl_pts_y_bottom = np.ones(F // 2)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C    # F x 2

    def _build_P(self, tar_size):
        h, w = tar_size
        I_r_grid_x = (np.arange(-w, w, 2) + 1.0) / w    # self.w
        I_r_grid_y = (np.arange(-h, h, 2) + 1.0) / h    # self.h
        P = np.stack(    # self.w x self.h x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        return P.reshape([-1, 2])    # n (= self.w x self.h) x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)    # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(    # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),    # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),    # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)    # 1 x F+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C    # F+3 x F+3

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]    # n (= tar_w x tar_h)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))    # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)    # 1 x F x 2
        P_diff = P_tile - C_tile    # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)    # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))    # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat    # n x F+3
