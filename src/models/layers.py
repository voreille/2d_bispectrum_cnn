from itertools import product

import tensorflow as tf
import numpy as np


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        self.fc_1 = tf.keras.layers.Dense(input_shape[-1] // self.ratio,
                                          activation="relu")
        self.fc_2 = tf.keras.layers.Dense(input_shape[-1],
                                          activation="sigmoid")

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return tf.multiply(inputs, x)


class BCHConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 kernel_size,
                 max_degree=6,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 use_bias=True,
                 bias_initializer="random_normal",
                 radial_profiles_sigma=0.25,
                 activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = max_degree
        self.n_radial_profiles = kernel_size // 2 + 1
        self.activation = tf.keras.layers.Activation(activation)
        self.conv_ch = CHConv2D(streams,
                                max_degree,
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
        # return ((self.degrees + 1) * (self.degrees + 2) * self.streams)
        return ((self.max_degree // 2 + 1)**2 * self.streams)

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n1 in range(self.max_degree // 2 + 1):
            for n2 in range(n1, self.max_degree // 2 + 1):
                bispectrum = (x[..., n1] * x[..., n2] *
                              tf.math.conj(x[..., n1 + n2]))
                if n1 == 0:
                    outputs.append(tf.math.real(bispectrum))
                else:
                    outputs.append(tf.math.real(bispectrum))
                    outputs.append(tf.math.imag(bispectrum))
        x = tf.concat(outputs, axis=-1)
        # x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        return self.activation(x)


class ECHConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 radial_profiles_sigma=0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = max_degree
        self.n_radial_profiles = kernel_size // 2 + 1
        self.conv_ch = CHConv2D(streams,
                                max_degree,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                initializer=initializer,
                                radial_profiles_sigma=radial_profiles_sigma,
                                **kwargs)

    @property
    def output_channels(self):
        return self.max_degree + 1

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n in range(self.max_degree + 1):
            energy = tf.math.real(x[..., n] * tf.math.conj(x[..., n]))
            outputs.append(energy)
        x = tf.concat(outputs, axis=-1)
        return tf.math.sqrt(x)


class BCHConv2DComplex(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 kernel_size,
                 max_degree=6,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 radial_profiles_sigma=0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = max_degree
        self.n_radial_profiles = kernel_size // 2 + 1
        self.conv_ch = CHConv2D(streams,
                                max_degree,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                initializer=initializer,
                                radial_profiles_sigma=radial_profiles_sigma,
                                **kwargs)

    @property
    def output_channels(self):
        return ((self.max_degree // 2 + 1) *
                (self.max_degree // 2 + 2) * self.streams) // 2

    @property
    def indices(self):
        indices = list()
        for n1 in range(self.max_degree // 2 + 1):
            for n2 in range(n1, self.max_degree // 2 + 1):
                indices.append(np.array([n1, n2]))

        return np.stack(indices, axis=-1)

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n1 in range(self.max_degree // 2 + 1):
            for n2 in range(n1, self.max_degree // 2 + 1):
                bispectrum = (x[..., n1] * x[..., n2] *
                              tf.math.conj(x[..., n1 + n2]))
                outputs.append(bispectrum)
        x = tf.concat(outputs, axis=-1)
        return x


class CHConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 max_degree,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 radial_profiles_sigma=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = max_degree
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
                self.max_degree + 1,
                self.n_radial_profiles,
            ),
            initializer=self.initializer,
            trainable=True,
        )

    @property
    def filters(self):
        weights = tf.complex(self.w, tf.zeros_like(self.w))
        return tf.reduce_sum(tf.multiply(weights, self.atoms), axis=-1)

    def call(self, inputs, training=None):
        filters = self.filters
        channels = tf.shape(filters)[2]
        filters = tf.reshape(filters, (
            self.kernel_size,
            self.kernel_size,
            channels,
            self.streams * (self.max_degree + 1),
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
            self.max_degree + 1,
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
                self.max_degree + 1,
                n_radial_profiles,
            ),
            dtype=np.csingle,
        )
        for i, n in product(range(n_radial_profiles),
                            range(self.max_degree + 1)):
            # atoms[:, :, 0, 0, n, i] = (np.exp(-0.5 * ((r - i) / sigma)**2) *
            #                            np.exp(theta * n * 1j))
            atoms[:, :, 0, 0, n, i] = (tri(r - i) * np.exp(theta * n * (-1j)))
        atoms[self.kernel_size // 2, self.kernel_size // 2, :, :, 1:, :] = 0
        norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1)))
        norm[norm == 0] = 1
        atoms = atoms / norm
        return tf.constant(atoms)


def tri(x):
    return np.where(np.abs(x) <= 1, np.where(x < 0, x + 1, 1 - x), 0)


class ResidualLayer2D(tf.keras.layers.Layer):
    def __init__(self, *args, activation='relu', **kwargs):
        super().__init__()
        self.filters = args[0]
        self.conv = tf.keras.layers.Conv2D(*args,
                                           **kwargs,
                                           activation=activation)
        self.activation = activation
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = None
        self.proj = None

    def build(self, input_shape):
        self.c_in = input_shape[1]
        if input_shape[1] != self.filters:
            self.proj = tf.keras.layers.Conv2D(self.filters,
                                               1,
                                               activation=self.activation)
            self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x


@tf.function
def conv2d_complex(input, filters, strides, **kwargs):
    out_channels = tf.shape(filters)[-1]
    filters_expanded = tf.concat(
        [tf.math.real(filters), tf.math.imag(filters)], axis=3)

    output = tf.nn.conv2d(input, filters_expanded, strides, **kwargs)
    return tf.complex(output[..., :out_channels], output[..., out_channels:])
