from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from scipy.ndimage import rotate

from src.data.monuseg import get_dataset, tf_random_rotate, tf_random_crop

ds = get_dataset()


def random_crop(image, segmentation, size=(256, 256), rotation=False):
    image_height, image_width, _ = image.shape
    radius = np.sqrt(size[0]**2 + size[1]**2) / 2
    if rotation:
        angle = np.random.uniform(-180, 180)
        dx = int((2 * radius - size[0]) // 2)
        dy = int((2 * radius - size[1]) // 2)
    else:
        dx, dy = 0, 0
    offset_height = np.random.randint(dx, high=image_height - size[0] - dx)
    offset_width = np.random.randint(dy, high=image_width - size[1] - dy)
    if rotation:
        image_cropped = image[offset_height - dx:offset_height + dx + size[0],
                              offset_width - dy:offset_width + dy + size[1]]
        seg_cropped = segmentation[offset_height - dx:offset_height + dx +
                                   size[0], offset_width - dy:offset_width +
                                   dy + size[1]]

        image_rotated = rotate(image_cropped, angle, reshape=False, order=1)
        seg_rotated = rotate(seg_cropped, angle, reshape=False, order=1)
        seg_rotated = tf.where(seg_rotated > 0.5, x=1.0, y=0.0)
        return (
            image_rotated[dx:dx + size[0], dy:dy + size[1]],
            seg_rotated[dx:dx + size[0], dy:dy + size[1]],
        )

    else:
        return (image[offset_height:offset_height + size[0],
                      offset_width:offset_width + size[1]],
                segmentation[offset_height:offset_height + size[0],
                             offset_width:offset_width + size[1]])


image, mask = next(ds.as_numpy_iterator())
image, mask = random_crop(image, mask, rotation=True)
print(f"yo la shape de liamg cest {image.shape}")