# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   created date: 2020/11/10
#   description:
#
#================================================================

from .model import OCRParallelDecoder as Model

from ..dataset import StrDataset
from ..label_converter import load_characters, Tokenizer, TextFilter
from ..common.metrics.accuracy import PerSampleAccuracy, normalize_text
from ..common import model_utils

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
    metrics = [PerSampleAccuracy(tokenizer)]

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
callbacks = model_utils.get_default_callbacks(
    params, additional_callbacks=additional_callbacks)


def evaluate(plot_attention=False, save_err_only=True):
    eval_db_names = glob.glob(os.path.join(params.eval_db_dir, '*'))
    logger.info(f'==========Begin evaluation==========')

    if save_err_only:
        save_dir = os.path.join(params.workspace, 'eval-error-attention_map')
    else:
        save_dir = os.path.join(params.workspace, 'eval-all-attention_map')

    logger.info(f'save images into {save_dir}')

    eval_steps = params.get('eval_steps', None)
    acc = {}
    for db_path in eval_db_names:
        db_name = os.path.split(db_path)[1]
        logger.info(f'\nEvaluate {db_name}')
        image_dir = os.path.join(save_dir, db_name)
        os.makedirs(image_dir, exist_ok=True)
        dataset.params.eval_db_names = [db_name]
        if plot_attention:
            dataset.params.batch_size = 1
        eval_dataloader = dataset.tfdataset('eval')

        total, err = 0, 0
        mean_time = 0
        for i, (img, label) in enumerate(eval_dataloader):
            if eval_steps is not None and i == eval_steps:
                break
            tar_real = label[:, 1:]

            if plot_attention:
                raise ValueError('Not implemented')
            else:
                t1 = time.time()
                logits, attention_weights = model(img, training=False)
                indices = tf.argmax(logits, axis=-1)
                t2 = time.time()
                if i != 0:  # skip calculate the first batch
                    mean_time += (t2 - t1 - mean_time) / i
                print(
                    f'mean inference time of a batch {params.batch_size} cost {mean_time}s'
                )

            y_pred = normalize_text(tokenizer.decode(indices)).numpy()
            y_true = normalize_text(tokenizer.decode(tar_real)).numpy()

            for j in range(len(y_pred)):
                total += 1

                img_name = f'{y_true[j].decode()}_{y_pred[j].decode()}.jpg'
                img_path = os.path.join(image_dir, img_name)

                if plot_attention and save_err_only and y_pred[j] != y_true[j]:
                    layer = 'decoder_layer4_block2'
                    raise ValueError('Not implemented')

                if y_pred[j] != y_true[j]:
                    err += 1
                    # tf.keras.preprocessing.image.save_img(img_path, img[j])
        acc[db_name] = (total - err) / total
        logger.info(f'Total:{total}, Error:{err}. acc:{(total-err)/total}')

    keys = [
        'IIIT5k_3000', 'SVT', 'IC03_867', 'IC03_860', 'IC13_857', 'IC13_1015',
        'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80'
    ]
    msg = ""
    for k in keys:
        v = acc[k]
        msg += "{:20}:{}\n".format(k, v)
    logger.info(msg)
    logger.info('=========end evaluation===========')


if params.mode == 'val':
    # model.evaluate(train_dataloader, steps=100)
    model.evaluate(val_dataloader)
elif params.mode == 'eval':
    plt_atn = params.get('plt_atn', False)
    save_err_only = params.get('plt_atn_err_only', False)
    evaluate(plt_atn, save_err_only)
else:
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
