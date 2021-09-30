import os
from functools import partial
from random import shuffle
import datetime

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow_addons.losses import sigmoid_focal_crossentropy

import click

from src.models.models import (Unet, LRIUnet, UnetLight, SpectUnetLight,
                               BispectUnetLight)
from src.models.loss import dice_coe_loss, dice_coe_metric
from src.data.drive import (get_dataset, tf_random_crop, tf_random_rotate,
                            tf_random_flip)


def loss_with_fl(y_true, y_pred):
    return dice_coe_loss(y_true, y_pred) + tf.reduce_mean(
        sigmoid_focal_crossentropy(
            y_true,
            y_pred,
            from_logits=False,
            alpha=0.25,
            gamma=2.0,
        ),
        axis=(1, 2))


def augment(image):
    image = tf.image.random_hue(image, 0.05)
    return tf.image.random_saturation(image, 0.8, 1.2)


@click.command()
@click.option('--model_name', type=click.STRING, default="BispectUnetLightDisk")
@click.option('--rotation/--no-rotation', default=True)
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('--focal_loss/--no-focal_loss', default=True)
@click.option('--cosine-decay/--no-cosine-decay', default=True)
@click.option('--cuda_core_id', type=click.STRING, default="1")
@click.option('--n_harmonics', type=click.INT, default=2)
def main(model_name, rotation, cuda_core_id, augmentation, focal_loss,
         cosine_decay, n_harmonics):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_core_id
    model_dict = {
        "Unet":
        Unet,
        "SpectUnet":
        partial(LRIUnet, kind="spectrum", n_harmonics=n_harmonics),
        "BispectUnet":
        partial(LRIUnet, kind="bispectrum", n_harmonics=n_harmonics),
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

    image_ids = [i for i in range(21, 41)]
    shuffle(image_ids)
    image_ids_val = image_ids[:5]
    image_ids_train = image_ids[5:]
    image_ids_train = image_ids_train[:1]

    ds_train = get_dataset(id_list=image_ids_train)
    ds_train = ds_train.cache().repeat(10)
    if rotation:
        ds_train = ds_train.map(tf_random_rotate).map(tf_random_flip)
    if augmentation:
        ds_train = ds_train.map(lambda x, y, z: (augment(x), y, z))
    ds_train = ds_train.map(tf_random_crop).map(lambda x, y, z: (x, z)).batch(
        2)

    ds_val = get_dataset(id_list=image_ids_val)
    f = lambda x: tf.image.resize_with_crop_or_pad(x, 592, 592)
    ds_val = ds_val.map(lambda x, y, z: (f(x), f(z))).cache().batch(1)

    val_images = list()
    val_segs = list()
    for image, seg in ds_val:
        val_images.append(image)
        val_segs.append(seg)
    val_images = tf.concat(val_images, 0)
    val_segs = tf.concat(val_segs, 0)

    dir_name = (model_name +
                f"__rotation_{rotation}__augmentation_{augmentation}__" +
                f"fl_{focal_loss}__" + f"cdecay__{cosine_decay}__" +
                f"nh_{n_harmonics}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = ("/home/valentin/python_wkspce/2d_bispectrum_cnn/logs/fit/" +
               dir_name)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    file_writer_image = tf.summary.create_file_writer(log_dir + '/images')

    def log_prediction(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        val_pred = model.predict(val_images)

        # Log the confusion matrix as an image summary.
        with file_writer_image.as_default():
            tf.summary.image("Validation images", val_images, step=epoch)
            tf.summary.image("Predictions", val_pred, step=epoch)
            tf.summary.image("GTs", val_segs, step=epoch)

    pred_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=log_prediction)

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
    model = model_dict[model_name](output_channels=1)

    # model = LRIUnet(output_channels=1)
    if focal_loss:
        loss = loss_with_fl
    else:
        loss = dice_coe_loss

    model.compile(
        loss=[loss],
        optimizer=optimizer,
        metrics=[dice_coe_metric, tf.keras.metrics.AUC()],
        run_eagerly=False,
    )
    model.fit(
        x=ds_train,
        epochs=1000,
        validation_data=ds_val,
        callbacks=[tensorboard_callback, pred_callback],
    )


if __name__ == '__main__':
    main()
