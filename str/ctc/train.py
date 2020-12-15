# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/10
#   description:
#
#================================================================

import os
import tensorflow as tf
keras = tf
from tfbp.param_helper import get_params, print_params
from tfbp.train import get_optimizer
from tfbp.logger import set_logger

from .model import CTCModel as Model
from ..dataset import StrDataset
from ..label_converter import load_characters, Tokenizer
from ..common.metrics.accuracy import PerSampleAccuracy
from ..common import model_utils

params = get_params(do_print=True)
logger = set_logger(log_to_file=os.path.join(params.workspace, 'log.txt'))

#load characters
params = load_characters(params)

# dataset
dataset = StrDataset(params)
train_dataloader = dataset.tfdataset('train')
val_dataloader = dataset.tfdataset('val')

# define model
if isinstance(params.gpu_id, (list, tuple)) and len(params.gpu_id) > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = Model(params)
    # define metrics
    metrics = [PerSampleAccuracy(Tokenizer(params.characters))]

optimizer = get_optimizer(params)
model.compile(
    optimizer=optimizer,
    # loss=loss,
    # loss_weights=loss_weight,
    metrics=metrics,
)

# load ckpt
with strategy.scope():
    initial_epoch = model_utils.load_weights(model, params)

# callbacks
callbacks = model_utils.get_default_callbacks(params, additional_callbacks=None)

if params.mode == 'val':
    model.evaluate(val_dataloader)
else:
    train_dataloader = train_dataloader.repeat()
    model.fit(
        train_dataloader,
        steps_per_epoch=1000,
        epochs=10000,
        validation_data=val_dataloader,
        callbacks=callbacks,
    )
