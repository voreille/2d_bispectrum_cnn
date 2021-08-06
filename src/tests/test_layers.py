import os
import random
import unittest
from itertools import product

from src.models.layers import conv2d_complex

import tensorflow as tf
import numpy as np
from numpy.random import seed
from scipy.ndimage import rotate
from scipy.signal import convolve2d

from src.models.layers import ECHConv2D, CHConv2DCompleteRadial

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)
tf.random.set_seed(2)


class TestComplexConv(unittest.TestCase):
    def _conv2d(self, image, filters):
        """
        rough implementation of convolution
        """
        (filter_height, filter_width, input_channels, streams,
         n_harmonics) = filters.shape
        batch_size, in_height, in_width, _ = image.shape

        output_channels = streams * n_harmonics
        filters_reshaped = np.reshape(filters, (
            filter_height,
            filter_width,
            input_channels,
            output_channels,
        ))

        output = np.zeros(
            (
                batch_size,
                in_height - filter_height + 1,
                in_width - filter_width + 1,
                input_channels,
                output_channels,
            ),
            dtype=np.csingle,
        )
        for batch, c_in, c_out in product(range(batch_size),
                                          range(input_channels),
                                          range(output_channels)):
            output[batch, :, :, c_in,
                   c_out] = convolve2d(image[batch, :, :, c_in],
                                       filters_reshaped[::-1, ::-1, c_in,
                                                        c_out],
                                       mode="valid")
        output = np.sum(output, axis=3)
        return np.reshape(
            output, (output.shape[0], output.shape[1], streams, n_harmonics))

    def test_conv(self):
        x = np.random.uniform(size=(200, 9, 9, 1))
        layer = CHConv2DCompleteRadial(
            1,
            4,
            9,
            initializer=tf.keras.initializers.Constant(1.0),
        )
        y = np.squeeze(layer(x).numpy())
        filters = layer.filters.numpy()
        y_np = np.squeeze(self._conv2d(x, filters))
        np.testing.assert_allclose(np.real(y), np.real(y_np), rtol=1e-3)
        np.testing.assert_allclose(np.imag(y), np.imag(y_np), rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
