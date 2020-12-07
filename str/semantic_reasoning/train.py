# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/10
#   description:
#
#================================================================

import os
import glob
import tensorflow as tf
keras = tf
from tfbp.param_helper import get_params, print_params
from tfbp.train import get_optimizer
from tfbp.logger import set_logger

from .model import SemanticReasonNet
from ..dataset import StrDataset
from ..label_converter import load_characters, Tokenizer
from ..common.metrics.accuracy import PerSampleAccuracy
from ..common import model_utils

params = get_params()
logger = set_logger(log_to_file=os.path.join(params.workspace, 'log.txt'))

#load characters
params = load_characters(params)

if isinstance(params.gpu_id, (list, tuple)) and len(params.gpu_id) > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

tokenizer = Tokenizer(params.characters)

print_params(params)

# define model
with strategy.scope():
    model = SemanticReasonNet(params)
    # define metrics
    if not params.with_semantic_reasoning:
        metrics = [PerSampleAccuracy(tokenizer)]
    else:
        metrics = {
            'output_1': PerSampleAccuracy(tokenizer),
            'output_2': PerSampleAccuracy(tokenizer),
            'output_3': PerSampleAccuracy(tokenizer)
        }

# define loss
if params.with_semantic_reasoning:
    loss = [keras.losses.SparseCategoricalCrossentropy(from_logits=True)] * 3
    loss_weight = [1.0, 0.15, 2.0]
else:
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = loss_fn
    loss_weight = None


def lr_schedule(epoch):
    if epoch < 45:
        return 1e-4
    elif epoch < 100:
        return 1e-5
    else:
        return 1e-6


optimizer = get_optimizer(params)
model.compile(
    optimizer=optimizer,
    loss=loss,
    loss_weights=loss_weight,
    metrics=metrics,
)

# load ckpt
with strategy.scope():
    initial_epoch = model_utils.load_weights(model, params)

# dataset
dataset = StrDataset(params)
train_dataloader = dataset.tfdataset('train')
val_dataloader = dataset.tfdataset('val')

# callbacks
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
additional_callbacks = [lr_schedule]
callbacks = model_utils.get_default_callbacks(
    params, additional_callbacks=additional_callbacks)


def evaluate():
    eval_db_names = glob.glob(os.path.join(params.eval_db_dir, '*'))
    logger.info(f'==========Begin evaluation==========')
    err_img_save_dir = os.path.join(params.workspace, 'eval_err_imgs')
    acc = {}
    for db_path in eval_db_names:
        db_name = os.path.split(db_path)[1]
        logger.info(f'\nEvaluate {db_name}')
        image_dir = os.path.join(err_img_save_dir, db_name)
        os.makedirs(image_dir, exist_ok=True)
        dataset.params.eval_db_names = [db_name]
        eval_dataloader = dataset.tfdataset('eval')

        total, err = 0, 0
        for img, label in eval_dataloader:
            if params.with_semantic_reasoning:
                logits1, logits2, logits3 = model.predict(img)
                logits = logits3
            else:
                logits = model.predict(img)

            indices = tf.argmax(logits, axis=-1)
            y_pred = tokenizer.decode(indices).numpy()
            y_true = tokenizer.decode(label).numpy()
            for j in range(len(y_pred)):
                total += 1
                if y_pred[j] != y_true[j]:
                    if '[UNK]' in y_true[j].decode('UTF-8'):
                        continue
                    err += 1
                    img_name = f'{y_true[j].decode()}_{y_pred[j].decode()}.jpg'
                    img_path = os.path.join(image_dir, img_name)
                    tf.keras.preprocessing.image.save_img(img_path, img[j])
        acc[db_name] = (total - err) / total
        logger.info(f'Total:{total}, Error:{err}. acc:{(total-err)/total}')

    print_params(acc)
    logger.info('=========end evaluation===========')


if params.mode == 'val':
    model.evaluate(val_dataloader)
elif params.mode == 'eval':
    evaluate()
else:
    steps_per_epoch = params.get('steps_per_epoch', 5000)
    model.evaluate(val_dataloader)
    model.fit(
        train_dataloader,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        epochs=1000,
        validation_data=val_dataloader,
        callbacks=callbacks,
    )
