# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/08
#   description:
#
#================================================================

import os
import re
import string
import numpy as np
import tensorflow as tf

PAD, UNK, SOS = '[PAD]', '[UNK]', '[SOS]'
PAD_ID, UNK_ID, SOS_ID = 0, 1, 2
RESERVED_TOKENS = [PAD, UNK, SOS]

EOS = '[EOS]'
EOS_ID = 3

from tfbp.logger import LOGGER
import re


class TextFilter(object):
    """filter texts according to characters"""

    def __init__(self, voc_type, characters):
        super(TextFilter, self).__init__()
        self.voc_type = voc_type
        if isinstance(characters, (list, tuple)):
            if characters[EOS_ID] == EOS:
                characters = characters[EOS_ID + 1:]
            elif characters[SOS_ID] == SOS:
                characters = characters[SOS_ID + 1:]
            else:
                characters = characters
            characters = ''.join(characters)
        self.pattern = f'[^{re.escape(characters)}]'
        LOGGER.info(f'regex_replace pattern:{self.pattern}')

    def __call__(self, text):
        if self.voc_type == 'LOWERCASE':
            text = tf.strings.lower(text)
        elif self.voc_type in ['ALLCASES', 'ALLCASES_SYMBOLS']:
            # text = tf.strings.regex_replace(text, self.pattern, '')
            text = text
        else:
            raise ValueError('TextFilter for voc_type: MANUAL, FILE is not implemented')
        return text


def load_characters(params):
    """load characters from `params.characters` to list. The loaded characters 
    will write  back to `params.characters`. The `params.dict_size` is 
    also set to `len(params.characters)`.

    Args:
        params (dict): has attribute `characters`.

    Returns: params.

    """
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'MANUAL', 'FILE']
    voc_type = params.voc_type
    if voc_type == 'LOWERCASE':
        characters = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        characters = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        characters = list(string.printable[:-6])
    elif voc_type == 'MANUAL':
        characters = list(params.characters)
    elif voc_type == 'FILE':
        characters = open(characters, 'r').read().splitlines()
    else:
        raise KeyError('voc_type must be one of {}'.format(types))
    with_eos = params.get('with_eos', False)
    reserved_tokens = RESERVED_TOKENS + [EOS] if with_eos else RESERVED_TOKENS

    characters = reserved_tokens + characters
    params.characters = characters
    params.dict_size = len(characters)
    LOGGER.info(f'dict_size: {params.dict_size}. first 100 chars: {params.characters[:100]}')
    return params


class Seq2seqLabelConverter(object):
    """convert label for seq2seq model"""

    def __init__(self, characters, max_n_chars):
        super(Seq2seqLabelConverter, self).__init__()
        self.characters = characters
        self.tokenizer = Tokenizer(characters)
        self.max_n_chars = max_n_chars

    def get_y_shape(self):
        return [self.max_n_chars]

    def encode(self, text):
        idx = self.tokenizer.encode(text)
        idx = tf.concat([[SOS_ID], idx, [EOS_ID]], axis=0)
        return idx
        # return {'text': text, 'idx': self.tokenizer.encode(text)}

    def decode(self, idxs):
        return self.tokenizer.decode(idxs)


class ParallelDecoderLabelConverter(object):
    """docstring for ParallelDecoderLabelConverter"""

    def __init__(self, characters, max_n_chars):
        super(ParallelDecoderLabelConverter, self).__init__()
        self.max_n_chars = max_n_chars
        self.tokenizer = Tokenizer(characters)

    def get_y_shape(self):
        return [self.max_n_chars]
        # return {'text': [], 'idx': [self.max_n_chars]}

    def encode(self, text):
        return self.tokenizer.encode(text)
        # return {'text': text, 'idx': self.tokenizer.encode(text)}

    def decode(self, idxs):
        return self.tokenizer.decode(idxs)


class Tokenizer(object):
    """text tokenizer"""

    def __init__(self, characters):
        super(Tokenizer, self).__init__()

        self.characters = [char.encode() for char in characters]
        self.with_eos = True if characters[EOS_ID] == EOS else False

        indexs = tf.cast(np.arange(len(self.characters)), dtype=tf.int32)

        self.idx2char_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(indexs, self.characters), PAD)
        self.char2idx_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.characters, indexs), UNK_ID)

    def encode(self, text):
        """encode text to idx.

        Args:
            text (Tensor of String): The label of image.

        Returns: List. The idx of the text.
        The idx of the text.

        """
        tokens = tf.strings.unicode_split(text, 'UTF-8')
        if isinstance(tokens, tf.RaggedTensor):
            tokens = tokens.to_tensor(default_value=PAD)
        indices = self.char2idx_table.lookup(tokens)
        return indices

    def decode(self, indices):
        """decode indices to text.

        Args:
            indices (Tensor): The indices of text

        Returns: text.

        """
        if not isinstance(indices, tf.Tensor):
            indices = tf.constant(indices)
        if indices.dtype != tf.int32:
            indices = tf.cast(indices, tf.int32)
        tokens = self.idx2char_table.lookup(indices)
        text = tf.strings.reduce_join(tokens, axis=-1)
        text = tf.strings.regex_replace(text, re.escape(PAD), '')    #delete PAD char
        if self.with_eos:
            text = tf.strings.regex_replace(text,
                                            re.escape(EOS) + '.*', '')    #delete EOS and the following chars
        return text
