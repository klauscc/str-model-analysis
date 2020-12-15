# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   created date: 2020/07/11
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

# load ckpt
with strategy.scope():
    initial_epoch = model_utils.load_weights(model, params)

dataset = StrDataset(params)


def evaluate(plot_attention=False, save_err_only=True):
    eval_db_names = glob.glob(os.path.join(params.eval_db_dir, '*'))
    logger.info(f'==========Begin evaluation==========')

    vis_layer = 'decoder_layer1_block2'
    if save_err_only:
        save_dir = os.path.join(params.workspace, 'eval-error-attention_map', vis_layer)
    else:
        save_dir = os.path.join(params.workspace, 'eval-all-attention_map', vis_layer)

    if plot_attention:
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
                indices, attention_weights = greedy_predict_one_image(model, dataset.params, img)
            else:
                t1 = time.time()
                output = model([img, None], training=False)
                t2 = time.time()
                if i != 0:    # skip calculate the first batch
                    mean_time += (t2 - t1 - mean_time) / i
                print(f'mean inference time of a batch {params.batch_size} cost {mean_time}s')
                indices = output['outputs']

            y_pred = normalize_text(tokenizer.decode(indices)).numpy()
            y_true = normalize_text(tokenizer.decode(tar_real)).numpy()

            for j in range(len(y_pred)):
                total += 1

                img_name = f'{y_true[j].decode()}_{y_pred[j].decode()}.jpg'
                img_path = os.path.join(image_dir, img_name)

                if plot_attention and (not save_err_only or y_pred[j] != y_true[j]):
                    plot_attention_weights(attention_weights,
                                            img,
                                            y_true[j].decode(),
                                            y_pred[j].decode(),
                                            img_path,
                                            layer=vis_layer)

                if y_pred[j] != y_true[j]:
                    err += 1
                    # tf.keras.preprocessing.image.save_img(img_path, img[j])
        acc[db_name] = (total - err) / total
        logger.info(f'Total:{total}, Error:{err}. acc:{(total-err)/total}')

    keys = [
        'IIIT5k_3000', 'SVT', 'IC03_867', 'IC03_860', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077',
        'SVTP', 'CUTE80'
    ]
    msg = ""
    for k in keys:
        v = acc[k]
        msg += "{:20}:{}\n".format(k, v)
    logger.info(msg)
    logger.info('=========end evaluation===========')


if __name__ == "__main__":
    plt_atn = params.get('plt_atn', False)
    save_err_only = params.get('plt_atn_err_only', False)
    evaluate(plt_atn, save_err_only)
