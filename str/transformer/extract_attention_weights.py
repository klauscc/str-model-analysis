# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: fengcheng
#   email: chengfeng2333@@gmail.com
#   created date: 2020/10/07
#   description:
#
#================================================================

import numpy as np
import tensorflow as tf
keras = tf.keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from ..label_converter import Tokenizer, SOS_ID, EOS_ID


def greedy_predict_one_image(model, params, image):
    """greedy predict one image for extracting attention_weights.

    Args:
        model : OCRTransformer. The model used to inference
        params : params.
        image : normalized image for predicting. Shape: [1, h, w, c] or [h, w, c] 

    Returns: TODO

    """
    if tf.rank(image) == 3:
        image = tf.expand_dims(image, 0)

    max_decode_length = params.max_n_chars - 1
    output = tf.ones([1, 1], dtype=tf.int32) * SOS_ID

    for i in range(max_decode_length):
        logits, attention_weights = model([image, output], training=False)
        #select the last word from the seq_len dimension
        logits = logits[:, -1:, :]  # (batch_size, 1, dict_size)
        predicted_id = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        if predicted_id == EOS_ID:
            return output[:, 1:], attention_weights

        output = tf.concat([output, predicted_id], axis=-1)
    return output[:, 1:], attention_weights


def overlay_atten2img(attention, image):
    """overlay attention maps to images

    Args:
        attention (TODO): TODO
        image (TODO): TODO

    Returns: TODO

    """
    pass


def plot_attention_weights(attention,
                           image,
                           sentence,
                           result,
                           save_path,
                           layer,
                           num_downsample=8):
    fig = plt.figure(figsize=(32, 9))
    fig.suptitle(f'gt:{sentence}.  predicted: {result}.  Layer:{layer}')

    attention = tf.squeeze(attention[layer],
                           axis=0)  # (head, n_chars, d_h*d_w)
    image = tf.squeeze(image, axis=0)
    image = image.numpy()
    image = np.array(image * 255, dtype=np.uint8)
    attention = attention.numpy()
    h, w, c = image.shape
    n_head, n_chars, _ = attention.shape
    if 'block2' in layer:
        attention = np.reshape(
            attention,
            [n_head, n_chars, h // num_downsample, w // num_downsample])

    ax = fig.add_subplot(n_head + 1, n_chars, 1)
    ax.imshow(image)
    ax.axis('off')
    ith = n_chars + 1
    for head in range(n_head):
        for i in range(n_chars):
            ax = fig.add_subplot(n_head + 1, n_chars, ith)
            ith += 1

            ith_attn_map = attention[head, i, :, :]
            ith_attn_map = ith_attn_map / np.max(ith_attn_map)
            ith_attn_map = ith_attn_map[:, :, np.newaxis]

            attn_map_img = np.array(ith_attn_map * 255, dtype=np.uint8)
            attn_map_img = cv2.resize(attn_map_img, (w, h),
                                      interpolation=cv2.INTER_CUBIC)
            # ax.matshow(attn_map_img, cmap='viridis')
            heatmap = cv2.applyColorMap(attn_map_img, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap, 0.4, image, 0.6, 0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            ax.imshow(overlay)
            ax.axis('off')

    # plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
