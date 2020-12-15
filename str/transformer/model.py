# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/21
#   description:
#
#================================================================

import functools
import numpy as np
import tensorflow as tf

from ..label_converter import Tokenizer, SOS_ID, EOS_ID
from ..common import network
from .mask import create_decoder_combined_mask, create_padding_mask, create_look_ahead_mask
from . import beam_search
from .losses import transformer_loss
from .modules import Decoder, PDDecoder, JitterSeqPredicton


class OCRTransformer(tf.keras.Model):
    """docstring for OCRTransformer"""

    def __init__(self, params):
        super(OCRTransformer, self).__init__()
        self.params = params

        input_shape = list(params.tar_img_size) + [3]

        self.backbone = network.build_backbone(input_shape,
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
        self.decoder = Decoder(params.num_decoder_layers, params.d_model, params.num_heads, params.dff,
                                params.dict_size, params.maximum_position_encoding)

        if params.with_parallel_visual_attention:
            if params.pd_method == 'semantic_reasoning':
                from ..semantic_reasoning.model import ParallelVisualAttentionModule
                self.parallel_visual_feature_aggregator = ParallelVisualAttentionModule(params.max_n_chars -
                                                                                        1,
                                                                                        params.d_model,
                                                                                        params.dict_size,
                                                                                        with_classifier=False)
            elif params.pd_method == 'dan_cam':
                from ..common.layers.parallel_decoding import CAMVisualFeatures
                h, w = params.tar_img_size
                d_scales = params.backbone_dowmsample_scales
                encoder_output_shape = (h // d_scales, w // d_scales, params.d_model)
                self.parallel_visual_feature_aggregator = CAMVisualFeatures(params.max_n_chars - 1,
                                                                            encoder_output_shape)

        self.classifier = tf.keras.layers.Dense(params.dict_size)

        self.loss_function = self.make_loss_function()
        self.unmasked_loss_function = functools.partial(transformer_loss,
                                                        smoothing=self.params.label_smooth,
                                                        vocab_size=self.params.dict_size,
                                                        mask_padding=False)
        self.strategy = tf.distribute.get_strategy()

    def call(self, data, training):
        inp, tar = data
        enc_mask = None

        encoder_outputs = self._encode(inp, training, enc_mask)
        if tar is not None:
            return self._decode(tar, encoder_outputs, training, enc_mask)
        else:    # predict
            return self._predict(encoder_outputs, training, enc_mask)

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
        encoder_outputs = self.encoder(feature, training=encoder_training, mask=None)
        return encoder_outputs

    def _decode(self, tar, encoder_outputs, training, enc_dec_mask=None):
        combined_mask = create_decoder_combined_mask(tar)

        if self.params.with_parallel_visual_attention:
            pd_features = self.parallel_visual_feature_aggregator(encoder_outputs, training)
        else:
            pd_features = None

        early_pd_feature = pd_features if self.params.pd_early_fusion and self.params.with_parallel_visual_attention else None
        dec_output, attention_weights = self.decoder(tar, encoder_outputs, training, combined_mask, None,
                                                        early_pd_feature)

        if self.params.pd_late_fusion:
            features = tf.concat([dec_output, pd_features], axis=-1)
        else:
            features = dec_output

        logits = self.classifier(features)    # (bs, max_n_chars, vocab_size)

        return logits, attention_weights

    def _predict(self, encoder_outputs, training, enc_dec_mask=None):
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]

        max_decode_length = self.params.max_n_chars - 1

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length, training)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.ones([batch_size], dtype=tf.int32) * SOS_ID

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        init_decode_length = (0)
        num_heads = self.params.num_heads
        dim_per_head = self.params.d_model // num_heads
        cache = {
            f"layer_{layer}": {
                "k": tf.zeros([batch_size, num_heads, init_decode_length, dim_per_head], dtype=tf.float32),
                "v": tf.zeros([batch_size, num_heads, init_decode_length, dim_per_head], dtype=tf.float32)
            } for layer in range(self.params.num_decoder_layers)
        }
        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs

        if self.params.with_parallel_visual_attention:
            pd_features = self.parallel_visual_feature_aggregator(encoder_outputs, training)
            cache['pd_features'] = pd_features

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(symbols_to_logits_fn=symbols_to_logits_fn,
                                                                initial_ids=initial_ids,
                                                                initial_cache=cache,
                                                                vocab_size=self.params.dict_size,
                                                                beam_size=self.params.beam_size,
                                                                alpha=self.params.alpha,
                                                                max_decode_length=max_decode_length,
                                                                eos_id=EOS_ID,
                                                                padded_decode=self.params.padded_decode,
                                                                dtype=tf.float32)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {
            "outputs": top_decoded_ids,
            "scores": top_scores,
        }

    def prepare_tar_inp_real(self, y, mode):
        tar_inp = y[:, :-1]
        tar_real = y[:, 1:]

        # if mode == 'train', perform label jitter
        if mode == 'train' and self.params.label_jitter > 0:
            bs = tf.shape(tar_inp)[0]
            inp_len = tf.shape(tar_inp)[1]
            jitter_values = tf.random.uniform([bs, inp_len],
                                                minval=EOS_ID + 1,
                                                maxval=self.params.dict_size,
                                                dtype=tf.int32)
            keep_probs = tf.concat(
                [
                    tf.ones([bs, 1]),    # do not jitter [SOS]
                    tf.random.uniform([bs, inp_len - 1], minval=0, maxval=1)
                ],
                axis=1)    # [bs, len]
            tar_inp = tf.where(keep_probs > self.params.label_jitter, tar_inp, jitter_values)

        return tar_inp, tar_real

    def train_step(self, data):
        x, y = data
        tar_inp, tar_real = self.prepare_tar_inp_real(y, mode='train')

        res = {}

        with tf.GradientTape() as tape:
            predictions, _ = self([x, tar_inp], True)
            per_replica_loss = self.loss_function(tar_real, predictions)

            scaled_loss = per_replica_loss / self.strategy.num_replicas_in_sync

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, predictions)

        res['loss'] = per_replica_loss
        res.update({m.name: m.result() for m in self.metrics})
        return res

    def test_step(self, data):
        x, y = data
        tar_inp, tar_real = self.prepare_tar_inp_real(y, mode='test')
        outputs = self([x, None], False)
        seq_predictions = outputs['seq_outputs']
        predictions = outputs['outputs']

        train_like_predictions, _ = self([x, tar_inp], False)

        # loss = self.unmasked_loss_function(tar_real, train_like_predictions)
        loss = self.loss_function(tar_real, train_like_predictions)

        self.compiled_metrics.update_state(tar_real, predictions)

        res = {'loss': loss}
        res.update({m.name: m.result() for m in self.metrics})
        return res

    def make_loss_function(self):
        loss_function = functools.partial(transformer_loss,
                                            smoothing=self.params.label_smooth,
                                            vocab_size=self.params.dict_size)
        self.loss_function = loss_function
        return loss_function

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""
        look_ahead_mask = create_look_ahead_mask(max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
          Args:
            ids: Current decoded sequences. int tensor with shape [batch_size *
              beam_size, i + 1].
            i: Loop index.
            cache: dictionary of values storing the encoder output, encoder-decoder
              attention bias, and previous decoder attention values.
          Returns:
            Tuple of
              (logits with shape [batch_size * beam_size, vocab_size],
               updated cache values)
          """
            decoder_input = ids[:, -1:]
            early_pd_feature = cache[
                'pd_features'] if self.params.pd_early_fusion and self.params.with_parallel_visual_attention else None
            dec_output, attention_weights = self.decoder(decoder_input,
                                                            cache['encoder_outputs'],
                                                            training,
                                                            look_ahead_mask,
                                                            None,
                                                            pd_features=early_pd_feature,
                                                            cache=cache,
                                                            loop_i=i)
            if self.params.with_parallel_visual_attention and self.params.pd_late_fusion:
                visual_feature = cache['pd_features'][:, i:i + 1, :]
                feature = tf.concat([dec_output, visual_feature], axis=-1)
            else:
                feature = dec_output
            logits = self.classifier(feature)    # (bs, 1, dict_size)
            logits = tf.squeeze(logits, axis=1)    # (bs, dict_size)
            return logits, cache

        return symbols_to_logits_fn
