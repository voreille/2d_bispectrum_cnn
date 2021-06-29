from itertools import product

import tensorflow as tf
import numpy as np


class BCHConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 degrees,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 use_bias=True,
                 bias_initializer="random_normal",
                 radial_profiles_sigma=0.5,
                 activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.degrees = degrees
        self.max_degree = 2 * degrees + 1
        self.n_radial_profiles = kernel_size // 2 + 1
        self.activation = tf.keras.layers.Activation(activation)
        self.conv_ch = CHConv2D(streams,
                                degrees,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                initializer=initializer,
                                radial_profiles_sigma=radial_profiles_sigma,
                                **kwargs)

        self.use_bias = use_bias
        if use_bias:
            self.bias = self.add_weight(shape=(self.output_channels, ),
                                        initializer=bias_initializer,
                                        trainable=True)
        else:
            self.bias = None

    @property
    def output_channels(self):
        return ((self.degrees + 1) * (self.degrees + 2) // 2 * self.streams)

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n1 in range(self.degrees + 1):
            for n2 in range(n1, self.degrees + 1):
                bispectrum = (x[..., n1] * x[..., n2] *
                              tf.math.conj(x[..., n1 + n2]))
                if (n1 + n2) % 2 == 0:
                    bispectrum = tf.math.real(bispectrum)
                else:
                    bispectrum = tf.math.imag(bispectrum)
                outputs.append(bispectrum)
        x = tf.concat(outputs, axis=-1)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        return self.activation(x)


class CHConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 degrees,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 radial_profiles_sigma=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.degrees = degrees
        self.max_degree = 2 * degrees + 1
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.n_radial_profiles = kernel_size // 2 + 1
        self.atoms = self._atoms(sigma=radial_profiles_sigma,
                                 n_radial_profiles=self.n_radial_profiles)
        self.initializer = initializer

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(
                1,
                1,
                input_shape[-1],
                self.streams,
                self.max_degree,
                self.n_radial_profiles,
            ),
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, inputs, training=None):
        weights = tf.complex(self.w, tf.zeros_like(self.w))
        filters = tf.reduce_sum(tf.multiply(weights, self.atoms), axis=-1)
        channels = tf.shape(inputs)[-1]
        filters = tf.reshape(filters, (
            self.kernel_size,
            self.kernel_size,
            channels,
            self.streams * self.max_degree,
        ))
        feature_maps = conv2d_complex(inputs,
                                      filters,
                                      self.strides,
                                      padding=self.padding)
        batch_size, height, width, _ = tf.shape(feature_maps)
        feature_maps = tf.reshape(feature_maps, (
            batch_size,
            height,
            width,
            self.streams,
            self.max_degree,
        ))
        return feature_maps

    def _atoms(self, sigma=0.5, n_radial_profiles=None):
        radius = (self.kernel_size - 1) // 2
        x_grid = np.arange(-radius, radius + 1, 1)
        x, y = np.meshgrid(x_grid, x_grid)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        if n_radial_profiles is None:
            n_radial_profiles = self.kernel_size // 2 + 1
        atoms = np.zeros(
            (
                self.kernel_size,
                self.kernel_size,
                1,
                1,
                2 * self.degrees + 1,
                n_radial_profiles,
            ),
            dtype=np.csingle,
        )
        for i, n in product(range(n_radial_profiles), range(2 * self.degrees)):
            atoms[:, :, 0, 0, n, i] = (np.exp(-0.5 * ((r - i) / sigma)**2) *
                                       np.exp(theta * n * 1j))
        norm = tf.math.sqrt(
            tf.reduce_sum(tf.multiply(tf.math.conj(atoms), atoms), axis=[0,
                                                                         1]))
        return tf.divide(atoms, norm)


@tf.function
def conv2d_complex(input, filters, strides, **kwargs):
    out_channels = tf.shape(filters)[-1]
    filters_expanded = tf.concat(
        [tf.math.real(filters), tf.math.imag(filters)], axis=3)

    output = tf.nn.conv2d(input, filters_expanded, strides, **kwargs)
    return tf.complex(output[..., :out_channels], output[..., out_channels:])