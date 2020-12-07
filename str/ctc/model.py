# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/13
#   description:
#
#================================================================
import tensorflow as tf
from ..label_converter import Tokenizer
from ..common import network


class CTCModel(tf.keras.Model):
    """docstring for CTCModel"""

    def __init__(self, params):
        super(CTCModel, self).__init__()
        self.params = params

        input_shape = list(params.tar_img_size) + [3]
        self.backbone = network.build_backbone(input_shape, params.d_model, params.bn_momentum)
        self.sa_encoder = network.SelfAttentionEncoder(params.d_model,
                                                       params.num_heads,
                                                       params.num_encoder_sa,
                                                       params.dff,
                                                       params.dropout_rate,
                                                       keep_shape=True)

        self.y_proj = tf.keras.layers.Dense(params.d_model)

        self.classifier = tf.keras.layers.Dense(params.dict_size)
        self.tokenizer = Tokenizer(params.characters)

    def call(self, inp, training):
        encoder_output = self.backbone(inp, training)
        encoder_output = self.sa_encoder(encoder_output, training)    # (bs, 8, 32, d_model)
        _, h, w, c = encoder_output.shape.as_list()
        encoder_output = tf.transpose(encoder_output, (0, 2, 3, 1))
        encoder_output = tf.reshape(encoder_output, (-1, w, h * c))
        encoder_output = self.y_proj(encoder_output)    # (bs, w, c)
        logits = self.classifier(encoder_output)
        return logits

    def make_train_function(self):

        strategy = tf.distribute.get_strategy()

        def train_function(iterator):
            data = next(iterator)
            outputs = self.distribute_strategy.run(self.train_step, args=(data,))
            per_replica_loss = outputs.pop('loss')
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            metrics = strategy.experimental_local_results(outputs)[0]
            metrics['loss'] = loss
            return metrics

        if not self.run_eagerly:
            train_function = tf.function(train_function, experimental_relax_shapes=True)

        self.train_function = train_function
        return self.train_function

    def make_test_function(self):

        strategy = tf.distribute.get_strategy()

        def test_function(iterator):
            data = next(iterator)
            outputs = self.distribute_strategy.run(self.test_step, args=(data,))
            per_replica_loss = outputs.pop('loss')
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            metrics = strategy.experimental_local_results(outputs)[0]
            metrics['loss'] = loss
            return metrics

        if not self.run_eagerly:
            test_function = tf.function(test_function, experimental_relax_shapes=True)

        self.test_function = test_function
        return self.test_function

    def train_step(self, data):
        x, labels = data
        label_length = tf.cast(tf.math.count_nonzero(labels, axis=-1), tf.int32)
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
            logits = tf.transpose(logits, (1, 0, 2))

            loss = tf.nn.ctc_loss(labels, logits, label_length, logit_length)
            total_loss = tf.reduce_mean(loss)
            scaled_loss = total_loss / tf.distribute.get_strategy().num_replicas_in_sync

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, logit_length)
        y_pred = tf.sparse.to_dense(decoded[0], default_value=0)
        self.compiled_metrics.update_state(labels, y_pred)

        res = {'loss': scaled_loss}
        res.update({m.name: m.result() for m in self.metrics})
        return res

    def test_step(self, data):
        x, labels = data

        label_length = tf.cast(tf.math.count_nonzero(labels, axis=-1), tf.int32)

        logits = self(x, training=False)
        logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(labels, logits, label_length, logit_length)
        total_loss = tf.reduce_mean(loss)
        scaled_loss = total_loss / tf.distribute.get_strategy().num_replicas_in_sync

        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, logit_length)
        y_pred = tf.sparse.to_dense(decoded[0], default_value=0)

        self.compiled_metrics.update_state(labels, y_pred)
        res = {'loss': scaled_loss}
        res.update({m.name: m.result() for m in self.metrics})
        return res
