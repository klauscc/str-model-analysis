# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   created date: 2020/07/07
#   description:
#
#================================================================

import os
import re
import sys
import lmdb
import hashlib
import string
import numpy as np
import tensorflow as tf
from functools import partial
from PIL import Image

import tensorflow_io as tfio
from tfbp.logger import LOGGER

from .utils import check_image_valid
from .augmenter import (AugmentorList, RandomResize, ImageEnhancement, ImagePerspectiveTransform)
from .label_converter import (ParallelDecoderLabelConverter, Seq2seqLabelConverter, load_characters,
                                RESERVED_TOKENS, TextFilter)


class StrDataset(object):
    """scene text recognition dataset."""

    def __init__(self, params):
        super(StrDataset, self).__init__()
        self.params = params

        scales = [64, 128, 192, 256] if params.multi_scales else [256]
        keep_aspect_ratio = params.get('keep_aspect_ratio', True)
        self.img_processor = AugmentorList([
            ImageEnhancement(params.with_image_transform, params.with_tps),
            RandomResize(scales=scales, max_h=params.tar_img_size[0], keep_aspect_ratio=keep_aspect_ratio),
        ])

        if params.label_converter == 'seq2seq':
            self.label_converter = Seq2seqLabelConverter(params.characters, params.max_n_chars)
        else:
            self.label_converter = ParallelDecoderLabelConverter(params.characters, params.max_n_chars)
        self.text_filter = TextFilter(params.voc_type, params.characters)

        img_shape = params.tar_img_size + [3]
        self.padded_shapes = (img_shape, self.label_converter.get_y_shape())

    def tfdataset(self, mode):
        """get dataset for model input.
        Returns: 

        """
        if mode == 'train':
            db_dir = self.params.db_dir
            db_names = self.params.db_names
            pick_ratios = self.params.db_pick_ratio
        elif mode == 'eval':
            db_dir = self.params.eval_db_dir
            db_names = self.params.eval_db_names
            pick_ratios = None
        else:
            db_dir = self.params.test_db_dir
            db_names = self.params.test_db_names
            pick_ratios = self.params.test_db_pick_ratio

        # datasets = [self.single_tfdataset(os.path.join(db_dir, db_name)) for db_name in db_names]
        datasets = [
            self.single_tfdataset_from_iotensor(os.path.join(db_dir, db_name), mode) for db_name in db_names
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            choice_dataset = self.generate_choice_dataset(pick_ratios, len(datasets))
            dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

        if mode == 'train':
            dataset = dataset.shuffle(self.params.buffer_size)

        #map
        sample_process_fn = partial(self.process_sample, mode=mode)
        dataset = dataset.map(sample_process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #batch
        dataset = dataset.padded_batch(self.params.batch_size, padded_shapes=self.padded_shapes)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def single_tfdataset_from_iotensor(self, db_path, mode):
        iotensor = tfio.experimental.IOTensor.from_lmdb(db_path)
        n_samples = iotensor.__getitem__('num-samples')
        LOGGER.info(f'db:{db_path} has {n_samples[0]} samples')
        n_samples = tf.strings.to_number(n_samples, out_type=tf.int64)[0]
        dataset = tf.data.Dataset.range(1, n_samples + 1)

        if mode == 'train':
            dataset = dataset.shuffle(n_samples)

        def get_image_label(idx):
            index = tf.strings.as_string(idx, width=9, fill='0')
            image = iotensor.__getitem__(tf.strings.join(['image-', index]))
            label = iotensor.__getitem__(tf.strings.join(['label-', index]))
            image = tf.squeeze(image)
            label = tf.squeeze(label)
            return image, label

        characters = ''.join(self.params.characters[3:])

        # pattern = f'[{characters}]*'
        # LOGGER.info(f'filter_pattern:{pattern}')

        def flter(image, label):
            if tf.strings.length(label) > self.params.max_n_chars:
                return False
            return True
            # if self.params.data_filtering_off:
            #     return True
            # else:
            #     if not self.params.sensitive:
            #         label = tf.strings.lower(label)
            # return tf.strings.regex_full_match(label, pattern)

        dataset = dataset.map(get_image_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(flter)
        if mode == 'train':
            dataset = dataset.repeat()
        return dataset

    def single_tfdataset(self, lmdb_path):
        """make a tfdataset from a single lmdb file.

        Args:
            lmdb_path (str): path to the lmdb file. 

        Returns: (tf.data.Dataset).
            The `dataset` return (imgbyte, label). dype: (tf.string,tf.string)

        """

        def get_lmdb_generator(root, params):
            lmdb_dataset = LmdbDataset(root, params, decode_img=False)
            indexs = np.arange(len(lmdb_dataset))
            # np.random.shuffle(indexs)
            for i in indexs:
                sample = lmdb_dataset[i]
                if sample is not None:
                    yield sample

        generator = partial(get_lmdb_generator, root=lmdb_path, params=self.params)
        dataset = tf.data.Dataset.from_generator(generator,
                                                    output_types=(tf.string, tf.string),
                                                    output_shapes=([], []))
        return dataset

    def process_sample(self, imgbyte, label, mode):
        """decode and transform image.

        Args:
            imgbyte (byte): bytes of image.
            label (str): label.

        Returns: (img, label)

        """

        def process_image(imgbyte, mode):
            channels = 3 if self.params.rgb else 1
            #when dtype == tf.float32, the img is between [0,1]
            img = tf.io.decode_image(imgbyte, channels=channels, dtype=tf.float32, expand_animations=False)
            if self.params.get('augment', True):
                img = self.img_processor.augment(img, mode)
            else:
                img = self.img_processor.augment(img, 'val')
            if self.params.img_zero_mean:
                img = (img - 0.5) / 0.5
            return img

        def process_label(label, mode):
            params = self.params
            label = self.text_filter(label)
            label = self.label_converter.encode(label)
            return label

        img = process_image(imgbyte, mode)
        label = process_label(label, mode)
        return img, label

    def generate_choice_dataset(self, pick_ratios, n_datasets):
        """TODO: Docstring for generate_choice_dataset.

        Args:
            pick_ratios (list): The ratio to pick from each dataset. If None, the each dataset will be evenly picked.
            n_datasets (int): the num of datasets.

        Returns: tf.data.Dataset. choice_dataset. 

        """
        if pick_ratios is None:
            pick_ratios = np.ones(n_datasets) / n_datasets
        else:
            pick_ratios = np.array(pick_ratios) / np.sum(pick_ratios)    # normalize to 1
        pickup_idx = []
        magnify = 100
        for i, ratio in enumerate(pick_ratios):
            pickup_idx += [i] * int(ratio * magnify)
        pickup_idx = np.array(pickup_idx)
        choice_dataset = tf.data.Dataset.from_tensor_slices(pickup_idx).shuffle(buffer_size=magnify).repeat()
        return choice_dataset


class LmdbDataset(tf.keras.utils.Sequence):

    def __init__(self, root, opt, decode_img=False):

        self.root = root
        self.opt = opt
        self.decode_img = decode_img
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1    # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.characters}]'
                    if len(label) > opt.max_n_chars:
                        continue
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            if not self.opt.sensitive:
                label = label.lower()
            if len(label) > self.opt.max_n_chars:
                label = label[:self.opt.max_n_chars]

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.characters}]'
            label = re.sub(out_of_char, '', label)

            if self.opt.decode_img:
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    if self.opt.rgb:
                        img = Image.open(buf).convert('RGB')    # for color image
                    else:
                        img = Image.open(buf).convert('L')

                except IOError:
                    print(f'Corrupted image for {index}')
                    # make dummy image and dummy label for corrupted image.
                    if self.opt.rgb:
                        img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                    else:
                        img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                    label = ''

                return (img, label.encode())
            else:
                imgbyte = bytes(imgbuf)
                if check_image_valid(imgbyte):
                    return (imgbyte, label.encode())
                else:
                    return None


if __name__ == "__main__":
    from tfbp.param_helper import get_params
    from PIL import Image

    default_params = {
        'rgb': True,
        'tar_img_size': (64, 256),
        'characters': '0123456789abcdefghijklmnopqrstuvwxyz',
        'voc_type': 'ALLCASES_SYMBOLS',
        'decode_img': False,
        'db_names': ['MJ/MJ_train', 'ST'],
        'db_pick_ratio': [0.5, 0.5],
        'db_cache_file': '/tmp/fengcheng/dataset/cache/ocr/MJ_ST',
        'test_db_dir': '/tmp/fengcheng/dataset/ocr/data_lmdb_release',
        'test_db_names': ['validation'],
        'test_db_pick_ratio': [1.],
        'buffer_size': 1000,
        'batch_size': 128,
        'num_parallel_calls': 8,
        'data_filtering_off': False,
        'max_n_chars': 25,
        'multi_scales': False,
        'label_converter': 'seq2seq',
        'with_eos': True,
        'img_zero_mean': False,
    }
    default_params.update({
        'db_dir': '/tmp/fengcheng/dataset/ocr/data_lmdb_release/training',
    # 'snapshot_dir': '/tmp/fengcheng/dataset/ocr/snapshot/ST',
    # 'db_names': ['ST'],
        'snapshot_dir': '/tmp/fengcheng/dataset/ocr/snapshot/border_replicate/MJ',
        'db_names': ['MJ/MJ_train'],
        'db_pick_ratio': [1.0],
    })
    params = get_params(default_params, do_print=True)
    params = load_characters(params)
    dataset = StrDataset(params)

    mode = 'train'
    if mode == 'val':
        params.snapshot_dir = params.snapshot_dir + '_val'

    os.makedirs(params.snapshot_dir, exist_ok=True)
    dataloader = dataset.tfdataset(mode=mode)
    save = True
    import re
    prog = re.compile('.*[\W]+.*')
    for i, sample in enumerate(dataloader):
        x, y = sample
        # orig_text = y['text'][0].numpy()
        # idx = y['idx'][0].numpy()
        decoded_text = dataset.label_converter.decode(y)
        decoded_text = decoded_text.numpy()
        y = y.numpy()
        for j in range(len(decoded_text)):
            text = decoded_text[j][5:].decode()
            if prog.match(text) is not None:
                print(y[j], text)

        if i % 100 == 0:
            print(i)

        # LOGGER.info(f'{i}, x:{x.shape},y:{y}, {decoded_text}')
        if save:
            x = x * 255.
            x = x.numpy()
            x = x.astype('uint8')
            for j in range(0, params.batch_size, 10):
                text = decoded_text[j][5:].decode()
                Image.fromarray(x[j], 'RGB').save(os.path.join(params.snapshot_dir, f'{i}_{j}_{text}.jpg'))
