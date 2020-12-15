# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   created date: 2020/11/10
#   description:
#
#================================================================

import functools
import numpy as np
import tensorflow as tf

from ..label_converter import Tokenizer, SOS_ID, EOS_ID
from ..common import network
from ..transformer.losses import transformer_loss
from .modules import PDDecoder
from ..common.layers.parallel_decoding import CAMVisualFeatures


class OCRParallelDecoder(tf.keras.Model):
    """docstring for OCRParallelDecoder"""
    def __init__(self, params):
        super(OCRParallelDecoder, self).__init__()
        self.params = params

        input_shape = list(params.tar_img_size) + [3]

        self.backbone = network.build_backbone(
            input_shape,
            params.d_model,
            params.bn_momentum,
            two_way_conv=params.two_way_conv,
            merge_mode=params.merge_mode)

        self.encoder = network.SelfAttentionEncoder(params.d_model,
                                                    params.num_heads,
                                                    params.num_encoder_sa,
                                                    params.dff,
                                                    params.dropout_rate,
                                                    keep_shape=False)

        embedded_inputs = False
        if params.decoder_input == 'cam':
            embedded_inputs = True
            h, w = params.tar_img_size
            d_scales = params.backbone_dowmsample_scales
            encoder_output_shape = (h // d_scales, w // d_scales,
                                    params.d_model)
            self.decoder_hint = CAMVisualFeatures(params.max_n_chars,
                                                  encoder_output_shape)

        self.pd_decoder = PDDecoder(params.num_decoder_layers,
                                    params.d_model,
                                    params.num_heads,
                                    params.dff,
                                    params.dict_size,
                                    params.maximum_position_encoding,
                                    embedded_inputs=embedded_inputs)

        self.classifier = tf.keras.layers.Dense(params.dict_size)
        self.loss_function = self.make_loss_function()
        self.strategy = tf.distribute.get_strategy()

    def make_loss_function(self):
        loss_function = functools.partial(transformer_loss,
                                          smoothing=self.params.label_smooth,
                                          vocab_size=self.params.dict_size)
        self.loss_function = loss_function
        return loss_function

    def call(self, x, training):
        bs = tf.shape(x)[0]
        enc_mask = None

        encoder_outputs = self._encode(x, training, enc_mask)

        if self.params.decoder_input == 'abs_pe':
            tar_inp = tf.tile(tf.expand_dims(tf.range(self.params.max_n_chars),
                                             axis=0),
                              multiples=[bs, 1])  # shape: (bs, max_n_chars)
        elif self.params.decoder_input == 'cam':
            tar_inp = self.decoder_hint(encoder_outputs, training)
        else:
            raise ValueError(
                f'params.decoder_input only accept "abs_pe" and "cam". Got {self.params.decoder_input}'
            )

        dec_output, attention_weights = self.pd_decoder(
            tar_inp, encoder_outputs, training)
        logits = self.classifier(dec_output)
        return logits, attention_weights

    def _encode(self, x, training, enc_mask=None):
        encode_training_mode = self.params.get('encode_training_mode', 'all')
        if encode_training_mode == 'all':
            backbone_training = training
            encoder_training = training
        elif encode_training_mode == 'encoder_only':
            backbone_training = False
            encoder_training = training
        elif encode_training_mode == 'none':
            backbone_training = False
            encoder_training = False
        feature = self.backbone(x, training=backbone_training)
        encoder_outputs = self.encoder(feature,
                                       training=encoder_training,
                                       mask=None)
        return encoder_outputs

    def train_step(self, data):
        x, y = data
        tar_real = y[:, 1:]  # disgard [SOS]
        res = {}

        with tf.GradientTape() as tape:
            logits, _ = self(x, True)
            per_replica_loss = self.loss_function(tar_real, logits)

            scaled_loss = per_replica_loss / self.strategy.num_replicas_in_sync

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, logits)

        res['loss'] = per_replica_loss
        res.update({m.name: m.result() for m in self.metrics})
        return res

    def test_step(self, data):
        x, y = data
        tar_real = y[:, 1:]  # disgard [SOS]
        res = {}
        logits, _ = self(x, False)

        self.compiled_metrics.update_state(tar_real, logits)

        res['loss'] = self.loss_function(tar_real, logits)
        res.update({m.name: m.result() for m in self.metrics})
        return res
