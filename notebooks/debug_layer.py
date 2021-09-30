from itertools import product

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from src.models.layers import BCHConv2D

IMAGE_SHAPE = (128, 128)

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


train = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train.batch(64)

sample_batch, sample_batch_mask = next(train_dataset.as_numpy_iterator())

layer = BCHConv2D(
    1,
    3,
    strides=1,
    initializer=tf.keras.initializers.Constant(1.0),
    proj_initializer=tf.keras.initializers.Constant(1.0),
    is_transpose=False,
    project=False,
    n_harmonics=2,
    radial_profile_type="disks",
)

y = layer(sample_batch)
print("yo")
