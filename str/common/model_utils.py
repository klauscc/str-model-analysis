# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/14
#   description:
#
#================================================================

import os
from tfbp.logger import LOGGER
import tensorflow as tf


def load_weights(model, params):
    """load checkpoint

    Args:
        model (keras.Model): The model to load weights.
        params (dict): parameters.

    Returns: (int). The epochs to resume training

    """
    ckpt_root = params.load_ckpt if hasattr(params, 'load_ckpt') else tf.train.latest_checkpoint(
        os.path.join(params.workspace, 'ckpt'))
    initial_epoch = 0
    if ckpt_root is not None and ckpt_root != "":
        model.load_weights(ckpt_root)
        filename = os.path.split(ckpt_root)[1]
        initial_epoch = int(filename.split('-')[1])
        LOGGER.info('Load checkpoint: {}'.format(ckpt_root))
    LOGGER.info('begin training from epoch: {}'.format(initial_epoch))
    return initial_epoch


def get_default_callbacks(params, additional_callbacks=None):
    """The default callbacks include: ModelCheckpoint, TensorBoard

    Args:
        additional_callbacks (List(Callback)): Additional callbacks
            appended to the default callbacks.

    Returns: List(Callback).

    """
    ckpt_root = os.path.join(params.workspace, 'ckpt')
    os.makedirs(ckpt_root, exist_ok=True)
    board_root = os.path.join(params.workspace, 'tensorboard')
    os.makedirs(board_root, exist_ok=True)

    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_root, 'ckpt-{epoch:02d}-val_loss_{val_loss:.2f}'))
    tfboard = tf.keras.callbacks.TensorBoard(log_dir=board_root, update_freq='batch')

    default_callbacks = [model_ckpt, tfboard]
    if additional_callbacks is not None:
        default_callbacks += additional_callbacks
    return default_callbacks


def insert_layer_nonseq(model,
                        layer_regex,
                        insert_layer_factory,
                        insert_layer_name=None,
                        position='after'):
    """
    for example:

    #replace batch_normalization with new layernorm
    ```
    backbone = insert_layer_nonseq(
        backbone, '.*_bn', lambda: tf.keras.layers.LayerNormalization(),
        position='replace')
    ```


    """

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update({layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [
            network_dict['new_output_tensor_of'][layer_aux]
            for layer_aux in network_dict['input_layers_of'][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            # if insert_layer_name:
            # new_layer.name = insert_layer_name
            # else:
            # new_layer.name = '{}_{}'.format(layer.name, new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name, layer.name,
                                                                position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    return keras.Model(inputs=model.inputs, outputs=model_outputs)
