from itertools import product

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

import tensorflow_datasets as tfds

from src.models.layers import ECHConv2D, BCHConv2DComplex

np.set_printoptions(precision=2, linewidth=150)

IMAGE_SHAPE = (129, 129)

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SHAPE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SHAPE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def random_rotate_image(image, label):
    # rotation = np.random.uniform(-180,180)
    rotation = 90
    image = rotate(image, rotation, reshape=False, order=1)
    label = rotate(label, rotation, reshape=False, order=1)
    return image, label, rotation


def tf_random_rotate_image(image, label):
    im_shape = image.shape
    label_shape = label.shape
    [image, label, rotation] = tf.py_function(
        random_rotate_image,
        [image, label],
        [tf.float32, tf.float32, tf.float32],
    )
    image.set_shape(im_shape)
    label.set_shape(label_shape)
    return image, label, rotation


train = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image)

train_rotated = train.map(tf_random_rotate_image,
                          num_parallel_calls=tf.data.AUTOTUNE)
test_rotated = test.map(tf_random_rotate_image)

train_dataset = train.batch(64)
test_dataset = test.batch(64)

train_dataset_rotated = train_rotated.batch(64)
test_dataset_rotated = test_rotated.batch(64)

sample_batch, sample_batch_mask = next(train_dataset.as_numpy_iterator())
sample_batch_rotated, sample_batch_mask_rotated, rotation = next(
    train_dataset_rotated.as_numpy_iterator())

layer = BCHConv2DComplex(1, 9, initializer=tf.keras.initializers.Constant(1.0))

y = layer(sample_batch)
y_rotated = layer(sample_batch_rotated)