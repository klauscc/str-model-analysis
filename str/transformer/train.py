# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/10
#   description:
#
#================================================================

from .model import OCRTransformer as Model
from ..dataset import StrDataset
from ..label_converter import load_characters, Tokenizer, TextFilter
from ..common.metrics.accuracy import PerSampleAccuracy, normalize_text
from ..common import model_utils
from .extract_attention_weights import greedy_predict_one_image, plot_attention_weights

import os
import glob
import time
import tensorflow as tf
keras = tf
from tfbp.param_helper import get_params, print_params
from tfbp.train import get_optimizer
from tfbp.logger import set_logger

params = get_params()
logger = set_logger(log_to_file=os.path.join(params.workspace, 'log.txt'))

#load characters
params = load_characters(params)

# dataset
dataset = StrDataset(params)
train_dataloader = dataset.tfdataset('train')
val_dataloader = dataset.tfdataset('val')

tokenizer = Tokenizer(params.characters)

# define model
if isinstance(params.gpu_id, (list, tuple)) and len(params.gpu_id) > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = Model(params)
    # define metrics
    if params.with_post_pd:
        metrics = {'outputs': [PerSampleAccuracy(tokenizer)], 'seq_outputs': [PerSampleAccuracy(tokenizer)]}
    else:
        metrics = [PerSampleAccuracy(tokenizer)]

if params.get('lr_transformer_schedule', False):
    from ..common.lr_scheduler import TransformerSchedule
    lr = TransformerSchedule(1.0, params.d_model, warmup_steps=params.warmup_steps)
    params.init_lr = lr

print_params(params)

optimizer = get_optimizer(params)
model.compile(
    optimizer=optimizer,
    metrics=metrics,
)

# load ckpt
with strategy.scope():
    initial_epoch = model_utils.load_weights(model, params)


def lr_schedule(epoch):
    if epoch < 45:
        return 1e-4
    elif epoch < 100:
        return 1e-5
    else:
        return 1e-6


# callbacks
if params.get('reduce_lr_on_plateau', False):
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                        factor=0.1,
                                                        patience=10,
                                                        verbose=1)
else:
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

additional_callbacks = [lr_schedule]
callbacks = model_utils.get_default_callbacks(params, additional_callbacks=additional_callbacks)

if params.mode == 'val':
    # model.evaluate(train_dataloader, steps=100)
    model.evaluate(val_dataloader)
else:    # 'train'
    steps_per_epoch = params.get('steps_per_epoch', 5000)
    # repeat is done in each single dataset.
    # train_dataloader = train_dataloader.repeat()
    model.evaluate(val_dataloader)
    model.fit(
        train_dataloader,
        steps_per_epoch=steps_per_epoch,
        epochs=1000,
        initial_epoch=initial_epoch,
        validation_data=val_dataloader,
        callbacks=callbacks,
    )
