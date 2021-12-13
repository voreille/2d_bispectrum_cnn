from itertools import product
import warnings

import tensorflow as tf
import numpy as np


def get_lri_conv2d(*args, kind="bispectrum", **kwargs):
    if kind == "bispectrum":
        return BCHConv2D(*args, **kwargs)
    elif kind == "spectrum":
        return ECHConv2D(*args, **kwargs)
    else:
        raise ValueError(f"The kind {kind} is not supported")


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
    def __init__(
            self,
            streams,
            kernel_size,
            n_harmonics=4,
            strides=1,
            padding='SAME',
            initializer="random_normal",
            use_bias=True,
            bias_initializer="zeros",
            radial_profile_type="complete",
            activation="relu",  # RECENTLY Changed 20211203 !!! 
            proj_activation="relu",
            proj_initializer="glorot_uniform",
            is_transpose=False,
            project=True,
            **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.n_harmonics = n_harmonics
        self.max_degree = n_harmonics - 1
        self._output_channels = None
        self._indices = None
        # if preactivation == "sqrt":
        #     self.preactivation = tf.math.sqrt
        # else:
        #     self.preactivation = None
        self.activation = tf.keras.activations.get(activation)

        self.conv_ch = CHConv2D.get(name=radial_profile_type)(
            streams,
            n_harmonics,
            kernel_size,
            strides=strides,
            padding=padding,
            initializer=initializer,
            is_transpose=is_transpose,
            **kwargs)

        if use_bias:
            self.bias = self.add_weight(
                shape=(self.output_channels, ),
                initializer=bias_initializer,
                trainable=True,
                name="bias_bchconv2d",
            )
        else:
            self.bias = None

        if project:
            self.proj_conv = tf.keras.layers.Conv2D(
                streams,
                1,
                kernel_initializer=proj_initializer,
                activation=proj_activation,
                padding="SAME")
        else:
            self.proj_conv = None

    @property
    def output_channels(self):
        if self._output_channels is None:
            output_channels = 0
            for n1, _ in self.indices:
                if n1 == 0:
                    output_channels += 1
                else:
                    output_channels += 2
            self._output_channels = output_channels * self.streams

        return self._output_channels

    @property
    def indices(self):
        if self._indices is None:
            self._indices = list()
            if self.max_degree == 0:
                self._indices.append((0, 0))
            else:
                for n1 in range(0, (self.max_degree - 1) // 2 + 1):
                    for n2 in range(n1, self.max_degree - n1 + 1):
                        self._indices.append((n1, n2))
        return self._indices

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n1, n2 in self.indices:
            bispectrum = (x[..., n1] * x[..., n2] *
                          tf.math.conj(x[..., n1 + n2]))
            if n1 == 0:
                outputs.append(tf.math.real(bispectrum))
            else:
                outputs.append(tf.math.real(bispectrum))
                outputs.append(tf.math.imag(bispectrum))
        x = tf.concat(outputs, axis=-1)
        x = tf.math.sign(x) * tf.math.log(1 + tf.math.abs(x))
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        return x


class ECHConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 kernel_size,
                 n_harmonics=4,
                 strides=1,
                 padding='SAME',
                 initializer="random_normal",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="complete",
                 preactivation="sqrt",
                 activation="linear",
                 proj_activation="relu",
                 proj_initializer="glorot_uniform",
                 is_transpose=False,
                 project=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.n_harmonics = n_harmonics
        # if preactivation == "sqrt":
        #     self.preactivation = tf.math.sqrt
        # else:
        # self.preactivation = None
        self.activation = tf.keras.activations.get(activation)

        self.conv_ch = CHConv2D.get(name=radial_profile_type)(
            streams,
            n_harmonics,
            kernel_size,
            strides=strides,
            padding=padding,
            initializer=initializer,
            is_transpose=is_transpose,
            **kwargs)

        if use_bias:
            self.bias = self.add_weight(
                shape=(self.output_channels, ),
                initializer=bias_initializer,
                trainable=True,
                name="bias_echconv2d",
            )
        else:
            self.bias = None

        if project:
            self.proj_conv = tf.keras.layers.Conv2D(
                streams,
                1,
                kernel_initializer=proj_initializer,
                activation=proj_activation,
                padding="SAME")
        else:
            self.proj_conv = None

    @property
    def output_channels(self):
        return self.n_harmonics * self.streams

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n in range(self.n_harmonics):
            energy = tf.math.real(x[..., n] * tf.math.conj(x[..., n]))
            outputs.append(energy)
        x = tf.concat(outputs, axis=-1)
        # if self.preactivation is not None:
        #     x = self.preactivation(x)
        x = tf.math.log(1 + x)
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        return x


class BCHConv2DComplex(tf.keras.layers.Layer):
    def __init__(self,
                 streams,
                 kernel_size,
                 n_harmonics=4,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 radial_profile_type="complete_radial",
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.max_degree = n_harmonics - 1
        self._indices = None
        self._output_channels = None

        if radial_profile_type == "complete":
            self.conv_ch = CHConv2DComplete(streams,
                                            n_harmonics,
                                            kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            initializer=initializer,
                                            **kwargs)
        elif radial_profile_type == "complete_radial":
            self.conv_ch = CHConv2DCompleteRadial(streams,
                                                  n_harmonics,
                                                  kernel_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  initializer=initializer,
                                                  **kwargs)

    @property
    def output_channels(self):
        if self._output_channels is None:
            self._output_channels = len(self.indices)
        return self._output_channels

    @property
    def indices(self):
        if self._indices is None:
            self._indices = list()
            for n1 in range(0, (self.max_degree - 1) // 2 + 1):
                for n2 in range(n1, self.max_degree - n1 + 1):
                    self._indices.append((n1, n2))
        return self._indices

    def call(self, inputs):
        x = self.conv_ch(inputs)
        outputs = list()
        for n1, n2 in self.indices:
            bispectrum = (x[..., n1] * x[..., n2] *
                          tf.math.conj(x[..., n1 + n2]))
            outputs.append(bispectrum)
        x = tf.concat(outputs, axis=-1)
        return x


class CHConv2D(tf.keras.layers.Layer):
    _registry = {}  # class var that store the different daughter

    def __init_subclass__(cls, name, **kwargs):
        cls.name = name
        CHConv2D._registry[name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def get(cls, name: str):
        return CHConv2D._registry[name]

    def __init__(self,
                 streams,
                 n_harmonics,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 is_transpose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.streams = streams
        self.n_harmonics = n_harmonics
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.atoms = self._atoms()
        self.initializer = initializer
        if is_transpose:
            self.convolution = conv2d_transpose_complex
        else:
            self.convolution = conv2d_complex

    def call(self, inputs, training=None):
        filters = self.filters
        channels = tf.shape(filters)[2]
        filters = tf.reshape(filters, (
            self.kernel_size,
            self.kernel_size,
            channels,
            self.streams * self.n_harmonics,
        ))
        feature_maps = self.convolution(inputs,
                                        filters,
                                        self.strides,
                                        padding=self.padding)

        # tf is too dumb for tf.shape(...)[:3]
        batch_size = tf.shape(feature_maps)[0]
        height = tf.shape(feature_maps)[1]
        width = tf.shape(feature_maps)[2]

        feature_maps = tf.reshape(feature_maps, (
            batch_size,
            height,
            width,
            self.streams,
            self.n_harmonics,
        ))
        return feature_maps

    def _atoms(self):
        raise NotImplementedError("It is an abstrac class")

    @property
    def filters(self):
        raise NotImplementedError("It is an abstrac class")


class CHConv2D2x2(CHConv2D, name="2x2"):
    def __init__(self,
                 streams,
                 n_harmonics,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 initializer="random_normal",
                 is_transpose=False,
                 **kwargs):
        super().__init__(streams,
                         n_harmonics,
                         2,
                         strides=strides,
                         padding=padding,
                         initializer=initializer,
                         is_transpose=is_transpose,
                         **kwargs)
        if kernel_size != 2:
            warnings.warn("The layer CHConv2D2x2 only accept kernel_size=2")

        self._n_radial_profiles = 1

    def _atoms(self):
        kernel_profile = np.ones((2, 2))
        x_grid = np.array([-1, 1])
        x, y = np.meshgrid(x_grid, x_grid)
        theta = np.arctan2(y, x)
        atoms = np.zeros(
            (2, 2, 1, 1, self.n_harmonics, 1),
            dtype=np.csingle,
        )
        for k in range(self.n_harmonics):
            atoms[:, :, 0, 0, k, 0] = kernel_profile * np.exp(1j * k * theta)

        norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1)))
        norm[norm == 0] = 1
        atoms = atoms / norm

        return tf.constant(atoms)

    @property
    def n_radial_profiles(self):
        return self._n_radial_profiles

    def build(self, input_shape):
        limit = limit_glorot(input_shape[-1], self.streams)
        self.w = self.add_weight(
            shape=(
                1,
                1,
                input_shape[-1],
                self.streams,
                self.n_harmonics,
                1,
            ),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="w_profile",
        )

    @property
    def filters(self):
        w = tf.complex(self.w, 0.0)
        return tf.reduce_sum(w * self.atoms, axis=-1)


class CHConv2DComplete(CHConv2D, name="complete"):
    def _atoms(self):
        kernel_profiles = self._compute_kernel_profiles()
        kernel_size, _, n_profiles = kernel_profiles.shape
        radius = (kernel_size - 1) // 2
        x_grid = np.arange(-radius, radius + 1, 1)
        x, y = np.meshgrid(x_grid, x_grid)
        theta = np.arctan2(y, x)
        atoms0 = np.zeros((kernel_size, kernel_size, 1, 1, 1),
                          dtype=np.csingle)
        atoms0[self.kernel_size // 2, self.kernel_size // 2, 0] = 0.5
        atoms = np.zeros(
            (kernel_size, kernel_size, 1, 1, self.n_harmonics, n_profiles),
            dtype=np.csingle,
        )
        for k, i in product(range(self.n_harmonics), range(n_profiles)):
            atoms[:, :, 0, 0, k,
                  i] = kernel_profiles[:, :, i] * np.exp(1j * k * theta)

        norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1)))
        norm[norm == 0] = 1
        atoms = atoms / norm

        return tf.constant(atoms0), tf.constant(atoms)

    def _compute_kernel_profiles(self):
        radius_max = self.kernel_size // 2
        n_profiles = radius_max**2 + radius_max
        x_grid = np.arange(-radius_max, radius_max + 1, 1)
        x, y = np.meshgrid(x_grid, x_grid)
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        r = np.sqrt(x**2 + y**2)
        kernel_profiles = np.zeros(
            (self.kernel_size, self.kernel_size, n_profiles))
        theta_shifts = [k * np.pi / 2 for k in range(4)]
        profile_counter = 0
        for i in range(1, radius_max + 1):
            n_pixels = 8 * i
            d_theta = theta[np.where(((np.abs(x) == i) | (np.abs(y) == i))
                                     & (r <= np.sqrt(2) * i))]
            d_theta.sort()
            d_theta = d_theta[:n_pixels // 4]
            for dt in d_theta:
                shifts = (dt + np.array(theta_shifts)) % (2 * np.pi)
                for t in shifts:
                    kernel_profiles[is_approx_equal(theta, t) &
                                    ((np.abs(x) == i) | (np.abs(y) == i))
                                    & (r <= np.sqrt(2) * i),
                                    profile_counter] = 1
                profile_counter += 1

        return kernel_profiles

    @property
    def n_radial_profiles(self):
        return self.atoms[1].shape[-1]

    def build(self, input_shape):
        limit = limit_glorot(input_shape[-1], self.streams)
        self.w = self.add_weight(
            shape=(
                1,
                1,
                input_shape[-1],
                self.streams,
                self.n_harmonics,
                self.n_radial_profiles,
            ),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="w_profile",
        )
        self.w0 = self.add_weight(
            shape=(
                1,
                1,
                input_shape[-1],
                self.streams,
                1,
            ),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="w0_profile",
        )

    @property
    def filters(self):
        atoms0, atoms = self.atoms
        w0 = tf.complex(self.w0, 0.0)
        w = tf.complex(self.w, 0.0)
        factor = tf.concat(
            [
                tf.ones((1, ), dtype=tf.complex64),
                tf.zeros((self.n_harmonics - 1, ), dtype=tf.complex64)
            ],
            axis=0,
        )
        factor = tf.reshape(factor, (1, 1, 1, 1, self.n_harmonics))
        return w0 * atoms0 * factor + tf.reduce_sum(w * atoms, axis=-1)


class CHConv2DOnDisks(CHConv2DComplete, name="disks"):
    def _compute_kernel_profiles(self):
        radius_max = self.kernel_size // 2
        x_grid = np.arange(-radius_max, radius_max + 1, 1)
        x, y = np.meshgrid(x_grid, x_grid)
        r = np.sqrt(x**2 + y**2)
        disks = np.zeros((self.kernel_size, self.kernel_size, radius_max))
        for i in range(0, radius_max):
            disks[..., i] = tri(r - (i + 1))
        return disks


class CHConv2DCompleteRadial(CHConv2DComplete, name="complete_radial"):
    def _compute_kernel_profiles(self):
        radius_max = self.kernel_size // 2
        x_grid = np.arange(-radius_max, radius_max + 1, 1)
        x, y = np.meshgrid(x_grid, x_grid)
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        kernel_profiles = list()
        disks = self._get_disks()
        theta_shifts = [k * np.pi / 2 for k in range(4)]
        for i in range(1, radius_max + 1):
            d_theta = theta[disks[..., i] != 0]
            n_pixels = d_theta.shape[0]
            d_theta.sort()
            d_theta = d_theta[:n_pixels // 4]
            for dt in d_theta:
                shifts = (dt + np.array(theta_shifts)) % (2 * np.pi)
                kernel_profile = np.zeros((self.kernel_size, self.kernel_size))
                for t in shifts:
                    kernel_profile[is_approx_equal(theta, t)
                                   & (disks[..., i] != 0)] = 1
                kernel_profiles.append(kernel_profile)

        return np.stack(kernel_profiles, axis=-1)

    def _get_disks(self):
        radius_max = self.kernel_size // 2
        n_profiles = radius_max + 1
        x_grid = np.arange(-radius_max, radius_max + 1, 1)
        x, y = np.meshgrid(x_grid, x_grid)
        r = np.sqrt(x**2 + y**2)
        disks = np.zeros((self.kernel_size, self.kernel_size, n_profiles))
        for i in range(1, n_profiles):
            disks[(r <= i) & (r > i - 1), i] = 1
        return disks


class CHConv2D_old(tf.keras.layers.Layer):
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
        limit = limit_glorot(input_shape[-1], self.streams)
        self.w = self.add_weight(
            shape=(
                1,
                1,
                input_shape[-1],
                self.streams,
                self.max_degree + 1,
                self.n_radial_profiles,
            ),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="w_profile",
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
        self.strides = kwargs.get("strides", 1)

    def build(self, input_shape):
        self.c_in = input_shape[1]
        if input_shape[-1] != self.filters:
            self.proj = tf.keras.layers.Conv2D(self.filters,
                                               1,
                                               activation=self.activation,
                                               strides=self.strides)
            self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x


class ResidualLRILayer2D(tf.keras.layers.Layer):
    def __init__(self, *args, kind="bispectrum", activation="relu", **kwargs):
        super().__init__()
        self.filters = args[0]
        self.conv = get_lri_conv2d(*args,
                                   **kwargs,
                                   activation=activation,
                                   kind=kind)
        self.strides = kwargs.get("strides", 1)
        self.activation = activation
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = None
        self.proj = None

    def build(self, input_shape):
        self.c_in = input_shape[1]
        if input_shape[-1] != self.filters:  # or self.strides == 2 ??
            self.proj = tf.keras.layers.Conv2D(
                self.filters,
                1,
                activation=self.activation,
                strides=self.strides,
                padding="SAME",
            )
            self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 trainable=True,
                 activation=None,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 mask=None,
                 **kwargs):
        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = self.add_weight(
            shape=(self.filters, ),
            initializer="zeros",
            trainable=True,
            name="bias_masked_conv2d",
        )
        self.activation = tf.keras.activations.get(activation)
        self.mask = tf.reshape(tf.Variable(mask, dtype=tf.float32),
                               mask.shape + (1, 1))
        if mask is None:
            raise ValueError("HEY!!, provide a mask")

    def build(self, input_shape):
        limit = limit_glorot(input_shape[-1], self.filters)
        self.w = self.add_weight(
            shape=(
                self.kernel_size,
                self.kernel_size,
                input_shape[-1],
                self.filters,
            ),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True,
            name="kernel_masked_conv2d",
        )

    @property
    def kernel(self):
        return self.mask * self.w

    def call(self, inputs, training=None):
        return self.activation(
            tf.nn.conv2d(
                inputs,
                self.kernel,
                self.strides,
                self.padding,
            ) + self.bias)


# @tf.function
def conv2d_complex(input, filters, strides, **kwargs):
    out_channels = tf.shape(filters)[-1]
    filters_expanded = tf.concat(
        [tf.math.real(filters), tf.math.imag(filters)], axis=3)

    output = tf.nn.conv2d(input, filters_expanded, strides, **kwargs)
    return tf.complex(output[..., :out_channels], output[..., out_channels:])


# @tf.function
def conv2d_transpose_complex(input, filters, strides, **kwargs):
    out_channels = tf.shape(filters)[-1]
    filters_expanded = tf.concat(
        [tf.math.real(filters), tf.math.imag(filters)], axis=3)

    output = conv2d_transpose(input, filters_expanded, strides, **kwargs)
    return tf.complex(output[..., :out_channels], output[..., out_channels:])


# @tf.function
def conv2d_transpose(input, filters, strides, **kwargs):
    filter_height, filter_width, _, out_channels = filters.get_shape().as_list(
    )
    batch_size = tf.shape(input)[0]
    in_height = tf.shape(input)[1]
    in_width = tf.shape(input)[2]
    if type(strides) is int:
        stride_h = strides
        stride_w = strides
    elif len(strides) == 2:
        stride_h, stride_w = strides

    padding = kwargs.get("padding", "SAME")
    if padding == 'VALID':
        output_size_h = (in_height - 1) * stride_h + filter_height
        output_size_w = (in_width - 1) * stride_w + filter_width
    elif padding == 'SAME':
        output_size_h = in_height * stride_h
        output_size_w = in_width * stride_w
    else:
        raise ValueError("unknown padding")
    output_shape = (batch_size, output_size_h, output_size_w, out_channels)

    return tf.nn.conv2d_transpose(input, tf.transpose(filters, (0, 1, 3, 2)),
                                  output_shape, strides, **kwargs)


def is_approx_equal(x, y, epsilon=1e-3):
    return np.abs(x - y) / (np.sqrt(np.abs(x) * np.abs(y)) + epsilon) < epsilon


def tri(x):
    return np.where(np.abs(x) <= 1, np.where(x < 0, x + 1, 1 - x), 0)


def limit_glorot(c_in, c_out):
    return np.sqrt(6 / (c_in + c_out))