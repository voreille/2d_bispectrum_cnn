from functools import partial

import tensorflow as tf
import numpy as np
from src.models.layers import get_lri_conv2d, MaskedConv2D


def get_model(model_name="Unet",
              output_channels=1,
              loss=None,
              metrics=None,
              n_harmonics=4,
              cosine_decay=True,
              run_eagerly=False,
              n_feature_maps=[8, 16, 32],
              lr=1e-3,
              last_activation="sigmoid",
              radial_profile_type="disks"):
    model_dict = {
        "Unet":
        Unet,
        "SpectUnet":
        partial(
            SpectUnet,
            n_harmonics=n_harmonics,
            radial_profile_type=radial_profile_type,
        ),
        "BispectUnet":
        partial(
            BispectUnet,
            n_harmonics=n_harmonics,
            radial_profile_type=radial_profile_type,
        ),
        "MaskedUnet":
        MaskedUnet,
    }

    if cosine_decay:
        lr_cosine = tf.keras.experimental.CosineDecayRestarts(
            lr,
            4500,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_cosine)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model = model_dict[model_name](
        output_channels=output_channels,
        last_activation=last_activation,
        n_feature_maps=n_feature_maps,
    )

    model.compile(
        loss=[loss],
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=run_eagerly,
    )
    return model


class UnetBase(tf.keras.Model):

    def __init__(self,
                 *args,
                 output_channels=3,
                 last_activation="softmax",
                 n_feature_maps=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if n_feature_maps is None:
            n_feature_maps = [8, 16, 32]
        self.kernel_size = 5
        self.conv_1 = self.get_conv_block(n_feature_maps[0])
        self.conv_2 = self.get_conv_block(n_feature_maps[1])
        self.conv_3 = self.get_conv_block(n_feature_maps[2])
        self.conv_4 = self.get_conv_block(n_feature_maps[1])
        self.conv_5 = self.get_conv_block(n_feature_maps[0])

        self.down_sampling_1 = tf.keras.layers.MaxPool2D()
        self.down_sampling_2 = tf.keras.layers.MaxPool2D()

        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation=last_activation,
                                   padding='SAME'),
        ])

        self.crop_1 = tf.keras.layers.Cropping2D(cropping=(4, 4))
        self.crop_2 = tf.keras.layers.Cropping2D(cropping=(16, 16))
        self.upsampling_1 = tf.keras.Sequential(
            [tf.keras.layers.UpSampling2D(interpolation="bilinear")], )
        self.upsampling_2 = tf.keras.Sequential(
            [tf.keras.layers.UpSampling2D(interpolation="bilinear")], )

    def get_conv_block(self, filters):
        raise NotImplementedError()

    def call(self, inputs, training=None):
        x1 = self.conv_1(inputs, training=training)
        x2 = self.conv_2(self.down_sampling_1(x1), training=training)
        x3 = self.conv_3(self.down_sampling_2(x2), training=training)

        x4 = self.conv_4(
            tf.concat([
                self.upsampling_1(x3),
                self.crop_1(x2),
            ], axis=-1),
            training=training,
        )

        x5 = self.conv_5(
            tf.concat([
                self.upsampling_2(x4),
                self.crop_2(x1),
            ], axis=-1),
            training=training,
        )

        return self.last(x5, training=training)


class Unet(UnetBase):

    def get_conv_block(self, filters, activation="relu"):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, self.kernel_size, padding="VALID"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
        ])


class MaskedUnet(UnetBase):

    def __init__(self,
                 *args,
                 output_channels=3,
                 last_activation="softmax",
                 n_feature_maps=[8, 16, 32],
                 **kwargs):

        self.mask = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ])

        super().__init__(*args,
                         output_channels=output_channels,
                         last_activation=last_activation,
                         n_feature_maps=n_feature_maps,
                         **kwargs)

    def get_conv_block(self, filters):
        return tf.keras.Sequential([
            MaskedConv2D(filters,
                         self.kernel_size,
                         mask=self.mask,
                         padding="VALID"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
        ])


class LRIUnetBase(UnetBase):

    def __init__(self,
                 *args,
                 output_channels=1,
                 radial_profile_type="disks",
                 n_harmonics=4,
                 **kwargs):
        self.n_harmonics = n_harmonics
        self.radial_profile_type = radial_profile_type
        # self.kind = None
        super().__init__(*args, output_channels=output_channels, **kwargs)

    def get_conv_block(self, filters):
        return tf.keras.Sequential([
            get_lri_conv2d(filters,
                           self.kernel_size,
                           padding="VALID",
                           radial_profile_type=self.radial_profile_type,
                           n_harmonics=self.n_harmonics,
                           kind=self.kind),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
        ])


class SpectUnet(LRIUnetBase):

    def __init__(self, *args, output_channels=1, **kwargs):
        self.kind = "spectrum"
        super().__init__(*args, output_channels=output_channels, **kwargs)


class BispectUnet(LRIUnetBase):

    def __init__(self, *args, output_channels=1, **kwargs):
        self.kind = "bispectrum"
        super().__init__(*args, output_channels=output_channels, **kwargs)
