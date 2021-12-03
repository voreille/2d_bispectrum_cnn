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
                 n_feature_maps=[8, 16, 32],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = 5
        self.down_block_1 = self.get_down_block(n_feature_maps[0])
        self.down_block_2 = self.get_down_block(n_feature_maps[1])
        self.down_block_3 = self.get_down_block(n_feature_maps[2],
                                                max_pool=False)
        self.up_block_1 = self.get_up_block(n_feature_maps[2])
        self.up_block_2 = self.get_up_block(n_feature_maps[1])
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 1, activation="softmax", padding='SAME'),
        ])

        self.crop_1 = tf.keras.layers.Cropping2D(cropping=(2, 2))
        self.crop_2 = tf.keras.layers.Cropping2D(cropping=(8, 8))
        self.upsampling_1 = tf.keras.layers.UpSampling2D(
            interpolation="bilinear")
        self.upsampling_2 = tf.keras.layers.UpSampling2D(
            interpolation="bilinear")

    def get_down_block(self, filters, max_pool=True):
        raise NotImplementedError()

    def get_up_block(self, filters, max_pool=True):
        raise NotImplementedError()

    def call(self, inputs, training=None):
        x1 = self.down_block_1(inputs, training=training)
        x2 = self.down_block_2(x1, training=training)
        x3 = self.down_block_3(x2, training=training)

        x4 = self.up_block_1(self.upsampling_1(
            tf.concat([x3, self.crop_1(x2)], axis=-1)),
                             training=training)

        x5 = self.up_block_2(self.upsampling_2(
            tf.concat([x4, self.crop_2(x1)], axis=-1)),
                             training=training)
        return self.last(x5, training=training)


class Unet(UnetBase):
    def get_down_block(self, filters, max_pool=True):
        if max_pool:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters,
                                       self.kernel_size,
                                       padding="VALID"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Activation("relu"),
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters,
                    self.kernel_size,
                    padding="VALID",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ])

    def get_up_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, self.kernel_size, padding="VALID"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
        ])


class MaskedUnet(UnetBase):
    def __init__(self,
                 *args,
                 output_channels=3,
                 last_activation="softmax",
                 n_feature_maps=[8, 16, 32],
                 **kwargs):
        super().__init__(*args,
                         output_channels=output_channels,
                         last_activation=last_activation,
                         n_feature_maps=n_feature_maps,
                         **kwargs)
        self.mask = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ])

    def get_down_block(self, filters, max_pool=True):
        if max_pool:
            return tf.keras.Sequential([
                MaskedConv2D(filters,
                             self.kernel_size,
                             mask=self.mask,
                             padding="VALID"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Activation("relu"),
            ])
        else:
            return tf.keras.Sequential([
                MaskedConv2D(
                    filters,
                    self.kernel_size,
                    padding="VALID",
                    mask=self.mask,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ])

    def get_up_block(self, filters):
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
                 radial_profile_type="complete",
                 n_harmonics=4,
                 **kwargs):
        self.n_harmonics = n_harmonics
        self.radial_profile_type = radial_profile_type
        # self.kind = None
        super().__init__(*args, output_channels=output_channels, **kwargs)

    def get_down_block(self, filters, max_pool=True):
        if max_pool:
            return tf.keras.Sequential([
                get_lri_conv2d(filters,
                               self.kernel_size,
                               padding="VALID",
                               radial_profile_type="disks",
                               n_harmonics=self.n_harmonics,
                               kind=self.kind),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Activation("relu"),
            ])
        else:
            return tf.keras.Sequential([
                get_lri_conv2d(filters,
                               self.kernel_size,
                               padding="VALID",
                               radial_profile_type="disks",
                               n_harmonics=self.n_harmonics,
                               kind=self.kind),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ])

    def get_up_block(self, filters):
        return tf.keras.Sequential([
            get_lri_conv2d(filters,
                           self.kernel_size,
                           padding="VALID",
                           radial_profile_type="disks",
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
