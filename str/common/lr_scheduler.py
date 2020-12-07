# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night. 
#   
#   author: fengcheng
#   email: chengfeng2333@@gmail.com
#   created date: 2020/12/06
#   description: 
#
#================================================================

import tensorflow as tf

class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, d_model, warmup_steps=4000):
        super(TransformerSchedule, self).__init__()

        self.init_lr = init_lr
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        learning_rate = self.init_lr
        learning_rate *= (self.d_model**-0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / self.warmup_steps)
        # Apply rsqrt decay
        learning_rate /= tf.sqrt(tf.maximum(step, self.warmup_steps))
        return tf.maximum(learning_rate, 1e-5)
