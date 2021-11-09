from functools import partial

import tensorflow as tf
import numpy as np
from src.models.layers import (ResidualLayer2D, ResidualLRILayer2D,
                               get_lri_conv2d)
from scipy.ndimage import watershed_ift
from skimage.measure import label


def get_model(
    model_name="UnetLight",
    output_channels=1,
    loss=None,
    metrics=None,
    n_harmonics=4,
    cosine_decay=True,
    run_eagerly=False,
    last_activation="sigmoid",
):
    model_dict = {
        "UnetLight":
        UnetLight,
        "SpectUnetLight":
        partial(SpectUnetLight, n_harmonics=n_harmonics),
        "BispectUnetLight":
        partial(BispectUnetLight, n_harmonics=n_harmonics),
        "BispectUnetLightDisk":
        partial(BispectUnetLight,
                n_harmonics=n_harmonics,
                radial_profile_type="disks"),
    }

    if cosine_decay:
        lr = tf.keras.experimental.CosineDecayRestarts(
            1e-3,
            4500,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = model_dict[model_name](output_channels=output_channels,
                                   last_activation=last_activation)

    model.compile(
        loss=[loss],
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=run_eagerly,
    )
    return model


class UnetLightBase(tf.keras.Model):
    def __init__(self,
                 *args,
                 output_channels=3,
                 last_activation="softmax",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(8),
            self.get_down_block(16),
            self.get_down_block(32),
        ]
        self.up_stack = [self.get_up_block(16), self.get_up_block(8)]

        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation=last_activation,
                                   padding='SAME'),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 5, padding='SAME'),
            ResidualLayer2D(filters, 5, padding='SAME'),
        ])

    def get_down_block(self, filters):
        raise NotImplementedError()

    def get_up_block(self, filters, n_conv=2):
        raise NotImplementedError()

    def predict_monuseg(self, x):
        y_pred = self.predict(x)
        y_pred_quantized = (y_pred > 0.5).astype(np.uint8)
        # y_pred_quantized = np.zeros_like(y_pred, dtype=np.uint8)
        # y_pred_quantized[..., 1] = (y_pred[..., 1] > 0.5).astype(np.uint8)
        # y_pred_quantized[..., 0] = (y_pred[..., 0] > 0.5).astype(np.uint8)
        # y_pred_quantized[..., 2] = (y_pred[..., 2] > 0.5).astype(np.uint8)
        batch_size = y_pred.shape[0]
        output = list()
        for s in range(batch_size):
            markers = label(y_pred_quantized[s, :, :, 0])
            markers[y_pred_quantized[s, :, :, 2] != 0] = -1
            out = watershed_ift(
                (y_pred_quantized[s, :, :, 1]).astype(np.uint8), markers)
            out[out == -1] = 0
            output.append(out)
        return np.stack(output, axis=0)

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)

        return self.last(x)


class LRIUnetLightBase(UnetLightBase):
    def __init__(self,
                 *args,
                 output_channels=1,
                 radial_profile_type="complete",
                 n_harmonics=4,
                 **kwargs):
        self.n_harmonics = n_harmonics
        self.radial_profile_type = radial_profile_type
        # self.kind = None
        super().__init__(*args, output_channels=output_channels, **kwargs)

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLRILayer2D(filters,
                               7,
                               radial_profile_type="complete",
                               kind=self.kind,
                               n_harmonics=self.n_harmonics,
                               padding='SAME'),
            ResidualLRILayer2D(filters,
                               3,
                               radial_profile_type=self.radial_profile_type,
                               kind=self.kind,
                               n_harmonics=self.n_harmonics,
                               padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            ResidualLRILayer2D(filters,
                               3,
                               strides=2,
                               radial_profile_type=self.radial_profile_type,
                               kind=self.kind,
                               n_harmonics=self.n_harmonics,
                               padding='SAME'),
            ResidualLRILayer2D(filters,
                               3,
                               kind=self.kind,
                               n_harmonics=self.n_harmonics,
                               radial_profile_type=self.radial_profile_type,
                               padding='SAME'),
        ])

    def get_up_block(self, filters, n_conv=2):
        return LRIUpBlockLight(filters,
                               n_conv=n_conv,
                               n_harmonics=self.n_harmonics,
                               kind=self.kind)


class SpectUnetLight(LRIUnetLightBase):
    def __init__(self, *args, output_channels=1, **kwargs):
        self.kind = "spectrum"
        super().__init__(*args, output_channels=output_channels, **kwargs)


class BispectUnetLight(LRIUnetLightBase):
    def __init__(self, *args, output_channels=1, **kwargs):
        self.kind = "bispectrum"
        super().__init__(*args, output_channels=output_channels, **kwargs)


class UnetLight(UnetLightBase):
    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 5, padding='SAME'),
            ResidualLayer2D(filters, 5, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 5, strides=2, padding='SAME'),
            ResidualLayer2D(filters, 5, padding='SAME'),
        ])

    def get_up_block(self, filters, n_conv=2):
        return UpBlockLight(filters, n_conv=n_conv)


class UpBlockLight(tf.keras.layers.Layer):
    def __init__(self, filters, *args, n_conv=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.Sequential()
        for _ in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv2D(filters,
                                       5,
                                       padding='SAME',
                                       activation='relu'), )
        self.trans_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                          2,
                                                          strides=(2, 2),
                                                          padding='SAME',
                                                          activation='relu')
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.trans_conv(x)
        x = self.concat([x, skip])
        return self.conv(x)


class LRIUpBlockLight(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 *args,
                 n_harmonics=4,
                 n_conv=2,
                 kind="bispectrum",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.Sequential()
        for k in range(n_conv):
            self.conv.add(
                get_lri_conv2d(filters,
                               3,
                               padding='SAME',
                               activation='relu',
                               n_harmonics=n_harmonics,
                               kind=kind), )
        self.trans_conv = get_lri_conv2d(filters,
                                         2,
                                         strides=(2, 2),
                                         padding='SAME',
                                         activation='relu',
                                         n_harmonics=n_harmonics,
                                         is_transpose=True,
                                         kind=kind,
                                         radial_profile_type="2x2")
        # self.trans_conv = tf.keras.layers.UpSampling2D(
        # interpolation="bilinear")
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.trans_conv(x)
        x = self.concat([x, skip])
        return self.conv(x)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters,
                                        size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_model(output_channels, input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels,
                                           3,
                                           strides=2,
                                           activation="sigmoid",
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
