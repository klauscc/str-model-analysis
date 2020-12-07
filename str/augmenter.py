# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/08
#   description:
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf

from .common.ops.utils import resize_keep_aspect_ratio, random_choice_1d


class ImageAugmentor(object):
    """docstring for ImageAugmentor"""

    def __init__(self):
        super(ImageAugmentor, self).__init__()

    def __call__(self, image, mode):
        return self.augment(image, mode)

    def augment(self, image):
        pass


class AugmentorList(object):
    """augment a list of augmentors"""

    def __init__(self, augmentors):
        super(AugmentorList, self).__init__()
        self.augmentors = augmentors

    def augment(self, image, mode):
        for augmentor in self.augmentors:
            image = augmentor(image, mode)
        return image


class RandomResize(ImageAugmentor):
    """random resize image.

    Args:
        scales (list):  The scales to reszie the image width. 
            Default is (64,128,196,256) 

    """

    def __init__(self, scales=(64, 128, 192, 256), max_h=64, keep_aspect_ratio=False):
        super(RandomResize, self).__init__()

        self.scales = scales
        self.max_h = max_h
        self.keep_aspect_ratio = keep_aspect_ratio

    def augment(self, image, mode):
        if mode == 'train':
            tar_w = random_choice_1d(self.scales)
        else:
            tar_w = self.scales[-1]

        if self.keep_aspect_ratio:
            resized_img = resize_keep_aspect_ratio(image, tar_w, max_h=self.max_h)
        else:
            resized_img = tf.image.resize(image, (self.max_h, tar_w))
        return resized_img


rand_scalar = lambda scale: scale * (np.random.rand() - 0.5) * 2


class ImagePerspectiveTransform(ImageAugmentor):
    """docstring for ImagePerspective"""

    def __init__(self, max_theta=20, with_tps=True):
        super(ImagePerspectiveTransform, self).__init__()
        self.max_theta = max_theta
        self.with_tps = with_tps

    def augment(self, image):
        return self.transform(image)

    @staticmethod
    def min_rectangle(box, relaxation=0):
        t = np.min(box[:, 1]) - relaxation
        l = np.min(box[:, 0]) - relaxation
        r = np.max(box[:, 0]) + relaxation
        b = np.max(box[:, 1]) + relaxation

        return np.array([[l, t], [r, t], [r, b], [l, b]])

    def transform(self, img):
        """TODO: Docstring for transform.

        Args:
            img (TODO): TODO

        Returns: TODO

        """
        h, w = img.shape[:2]
        pad_width = max(h, w) * 2 // 3
        new_img = np.pad(img, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], 'linear_ramp')
        box = np.array([[pad_width, pad_width], [pad_width + w, pad_width], [pad_width + w, pad_width + h],
                        [pad_width, pad_width + h]])

        try:
            r = np.random.rand()
            if self.with_tps:
                if r < 0.6:    # tps transform
                    new_img, box = self._tps_transform(new_img, box, pad_width)
                elif r < 0.8:
                    new_img, box = self._parallelogram_transform(new_img, box)
                else:
                    new_img, box = self._random_transform(new_img, box)
            else:
                if r < 0.5:
                    new_img, box = self._parallelogram_transform(new_img, box)
                else:
                    new_img, box = self._random_transform(new_img, box)

            box = self.min_rectangle(box)
            box[:, 0] = np.clip(box[:, 0], 0, pad_width * 2 + w)
            box[:, 1] = np.clip(box[:, 1], 0, pad_width * 2 + h)
            l, u, r, b = np.array([
                np.min(box[:, 0]), np.min(box[:, 1]),
                np.max(box[:, 0]), np.max(box[:, 1])
            ],
                                    dtype=np.int32)
            if l > r - 4 or u > b - 4:
                return img
            else:
                return new_img[u:b, l:r, :]

        except Exception as e:
            print(e)
            return img

    def get_rotate_box(self, img, box):
        delta = np.random.uniform(-1 * self.max_theta, self.max_theta)
        height, width = img.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), delta, 1)
        new_box = cv2.transform(box[np.newaxis, :, :], M)
        return new_box[0, :, :]

    def _tps_transform(self, img, box, pad_width):
        l, u, r, b = np.array(
            [np.min(box[:, 0]), np.min(box[:, 1]),
                np.max(box[:, 0]), np.max(box[:, 1])], dtype=np.int32)
        N = 5
        circle = False

        matches = []
        for i in range(1, N + 1):
            matches.append(cv2.DMatch(i, i, 0))

        jitter_size = pad_width // 4

        def get_src_tar_shape(N):
            src_points = []
            dx = (r - l) // (N - 1)
            for i in range(N):
                src_points.append((l + dx * i, u))
                src_points.append((l + dx * i, b))

            source_shape = np.array(src_points, np.int32)
            source_shape = np.reshape(source_shape, (1, 2 * N, 2))

            jitters = (np.random.rand(1, 2 * N, 2) - 0.5) * jitter_size
            target_shape = np.copy(source_shape) - jitters
            target_shape = target_shape.astype(np.int32)
            return source_shape, target_shape

        source_shape, target_shape = get_src_tar_shape(N)

        if circle:
            for i in range(2 * N):
                cv2.circle(img, tuple(source_shape[0, i, :]), 1, (255, 0, 0), 5)

        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(target_shape, source_shape, matches)
        new_img = tps.warpImage(img)

        if circle:
            for i in range(2 * N):
                cv2.circle(new_img, tuple(target_shape[0, i, :]), 1, (0, 255, 0), 2)

        bbox = self.min_rectangle(np.squeeze(target_shape, axis=0), relaxation=jitter_size // 2)
        return new_img, bbox

    def _random_transform(self, img, box):
        box = np.array(box, dtype="float32")

        t = np.min(box[:, 1])
        l = np.min(box[:, 0])
        r = np.max(box[:, 0])
        b = np.max(box[:, 1])
        h = b - t
        w = r - l
        jitter_fn = lambda scale: scale * (np.random.rand(4) - 0.5) * 2
        jitter_y = h * jitter_fn(0.2)
        jitter_x = h * jitter_fn(0.2)

        new_box = box.copy()
        new_box[:, 0] = box[:, 0] - jitter_x
        new_box[:, 1] = box[:, 1] - jitter_y

        new_box = self.get_rotate_box(img, new_box)

        M = cv2.getPerspectiveTransform(box, new_box)
        wrapped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        return wrapped_img, new_box

    def _parallelogram_transform(self, img, box):
        box = np.array(box, dtype="float32")

        t = np.min(box[:, 1])
        l = np.min(box[:, 0])
        r = np.max(box[:, 0])
        b = np.max(box[:, 1])
        h = b - t
        w = r - l
        shift_y = h * rand_scalar(0.5)
        shift_y_narrow = h * rand_scalar(0.1)
        change_left_y = np.random.rand() < 0.5
        idx = [0, 3] if change_left_y else [1, 2]
        new_box = box.copy()
        new_box[idx, 1] += shift_y
        new_box[idx[1], 1] += shift_y_narrow

        shift_x = [h * rand_scalar(0.1) for i in range(4)]
        new_box[:, 0] += shift_x

        new_box = self.get_rotate_box(img, new_box)

        M = cv2.getPerspectiveTransform(box, new_box)
        wrapped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        return wrapped_img, new_box


def random_aug_fn(fn, im, delta_range):
    v = np.random.uniform(delta_range[0], delta_range[1])
    return fn(im, v)


def random_lines(img, max_lines=20):
    img = np.copy(img)
    h, w = img.shape[:2]
    n_lines = np.random.randint(1, max_lines)

    def random_point():
        return (np.random.randint(w), np.random.randint(h))

    for i in range(n_lines):
        start_point = random_point()
        end_point = random_point()
        color = np.random.randint(0, 255, size=3)
        img = cv2.line(img, start_point, end_point, (int(color[0]), int(color[1]), int(color[2])), 1)
    return img


class ImageEnhancement(ImageAugmentor):
    """enhance image such as adjust image quality, rotate."""

    def __init__(
            self,
            with_image_transform=True,
            with_tps=True,
            no_aug_prob=0.2,
            random_gaussian_prob=0.3,
            random_transform_prob=0.5,
            rotate_max_theta=20,
            random_add_line_prob=0.1,
            max_lines=10,
            random_brightness_prob=0.2,
            brightness_max_delta=0.2,
            random_contrast_prob=0.2,
            contrast_range=(0.8, 1.2),
            random_hue_prob=0.3,
            hue_max_delta=0.2,
            random_jpeg_quality_prob=0.3,
            jpeg_quality_range=(70, 100),
            random_saturation_prob=0.3,
            saturation_range=(0.8, 1.2),
            random_illuminant_prob=0.4,
            illuminant_change_range=(0.1, 0.5),
            random_downsample_prob=0.3,
    ):
        super(ImageEnhancement, self).__init__()

        self.no_aug_prob = no_aug_prob
        self.random_gaussian_prob = random_gaussian_prob
        self.random_brightness_prob = random_brightness_prob
        self.brightness_max_delta = brightness_max_delta
        self.random_contrast_prob = random_contrast_prob
        self.contrast_range = contrast_range
        self.random_hue_prob = random_hue_prob
        self.hue_max_delta = hue_max_delta
        self.random_jpeg_quality_prob = random_jpeg_quality_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.random_saturation_prob = random_saturation_prob
        self.saturation_range = saturation_range

        self.random_transform_prob = random_transform_prob
        self.rotate_max_theta = rotate_max_theta

        self.random_illuminant_prob = random_illuminant_prob
        self.illuminant_change_range = illuminant_change_range

        self.random_downsample_prob = random_downsample_prob

        self.random_add_line_prob = random_add_line_prob
        self.max_lines = max_lines

        self.with_image_transform = with_image_transform
        self.with_tps = with_tps

        self.perspective_transform = ImagePerspectiveTransform(max_theta=rotate_max_theta, with_tps=with_tps)

    def py_augmenter_wrapper(self, img):

        def func(img):
            if self.with_image_transform:
                if np.random.rand() < self.random_transform_prob:
                    img = self.perspective_transform.augment(img)
            if np.random.rand() < self.random_add_line_prob:
                img = random_lines(img, max_lines=self.max_lines)
            return img

        image = tf.py_function(func, inp=[img], Tout=tf.float32)
        image.set_shape([None, None, None])
        return image

    @staticmethod
    def change_illuminant(im, ratio=0.5):
        h = tf.shape(im)[0]
        w = tf.shape(im)[1]
        c = tf.shape(im)[2]
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        center_x = tf.random.uniform([1, 1, c], maxval=w)
        center_y = tf.random.uniform([1, 1, c], maxval=h)
        max_d = tf.sqrt(
            tf.square(tf.maximum(w - center_x, center_x)) + tf.square(tf.maximum(h - center_y, center_y)))
        iv, jv = tf.meshgrid(tf.range(tf.cast(w, tf.int32)), tf.range(tf.cast(h, tf.int32)))
        iv = tf.cast(iv[:, :, tf.newaxis], tf.float32)
        jv = tf.cast(jv[:, :, tf.newaxis], tf.float32)
        dx = tf.square(jv - center_x)
        dy = tf.square(iv - center_y)
        d = tf.sqrt(dx + dy)    #[h,w,c]
        percentage = d * -1.0 * ratio / max_d + 1.0    #[h,w,c]
        im = im * percentage
        return im

    @staticmethod
    def image_downsample(im, ratio):
        h = tf.shape(im)[0]
        w = tf.shape(im)[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)
        new_h = tf.maximum(new_h, 5)
        new_w = tf.maximum(new_w, 5)
        return tf.image.resize(im, [new_h, new_w])

    def augment(self, image, mode):
        if mode != 'train':
            return image
        if tf.random.uniform([]) < self.no_aug_prob:
            return image

        image = self.py_augmenter_wrapper(image)

        if tf.random.uniform([]) < self.random_gaussian_prob:
            image = tf.clip_by_value(
                image +
                tf.random.uniform([], 0, 0.2) * tf.random.normal(tf.shape(image), mean=0.0, stddev=0.5), 0.,
                1.)

        if tf.random.uniform([]) < self.random_brightness_prob:
            image = tf.image.random_brightness(image, self.brightness_max_delta)
        if tf.random.uniform([]) < self.random_contrast_prob:
            image = tf.image.random_contrast(image, *self.contrast_range)
        if tf.random.uniform([]) < self.random_hue_prob:
            image = tf.image.random_hue(image, self.hue_max_delta)
        if tf.random.uniform([]) < self.random_jpeg_quality_prob:
            image = tf.image.random_jpeg_quality(image, *self.jpeg_quality_range)
        if tf.random.uniform([]) < self.random_saturation_prob:
            image = tf.image.random_saturation(image, *self.saturation_range)
        if tf.random.uniform([]) < self.random_illuminant_prob:
            ratio = tf.random.uniform([],
                                        minval=self.illuminant_change_range[0],
                                        maxval=self.illuminant_change_range[1])
            image = self.change_illuminant(image, ratio)

        if tf.random.uniform([]) < self.random_downsample_prob:
            ratio = tf.random.uniform([], minval=0.5, maxval=0.8)
            image = self.image_downsample(image, ratio)

        return image
