import os
from pathlib import Path
from random import shuffle
import datetime
import json

import tensorflow as tf
import click
from tensorflow.python.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score

from src.models.models import get_model
from src.models.loss import loss_with_fl, dice_coe_metric, dice_coe_loss
from src.data.drive import (get_dataset, tf_random_crop, tf_random_rotate,
                            tf_random_flip)

image_dir = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Images_normalized"


def augment(image):
    image = tf.image.random_hue(image, 0.05)
    return tf.image.random_saturation(image, 0.8, 1.2)


@click.command()
@click.option('--model_name', type=click.STRING, default="UnetLight")
@click.option('--rotation/--no-rotation', default=True)
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('--focal_loss/--no-focal_loss', default=True)
@click.option('--cosine-decay/--no-cosine-decay', default=False)
@click.option('--cuda_core_id', type=click.STRING, default="3")
@click.option('--n_harmonics', type=click.INT, default=4)
@click.option('--n_train', type=click.INT, default=-1)
def main(model_name, rotation, cuda_core_id, augmentation, focal_loss,
         cosine_decay, n_harmonics, n_train):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_core_id

    image_ids = [f.stem for f in Path(image_dir).rglob("*.tiff")]
    shuffle(image_ids)
    image_ids_val = image_ids[:5]
    image_ids_train = image_ids[5:]
    image_ids_train = image_ids_train[:n_train]

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
                f"nh_{n_harmonics}__" + f"n_train_{n_train}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = (
        "/home/valentin/python_wkspce/2d_bispectrum_cnn/logs/fit_monuseg/" +
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

    if focal_loss:
        loss = loss_with_fl
    else:
        loss = dice_coe_loss

    model = get_model(model_name=model_name,
                      loss=loss,
                      metrics=[dice_coe_metric, AUC()],
                      n_harmonics=n_harmonics,
                      cosine_decay=cosine_decay,
                      run_eagerly=False)
    model.fit(
        x=ds_train,
        epochs=1000,
        validation_data=ds_val,
        callbacks=[tensorboard_callback, pred_callback],
    )


if __name__ == '__main__':
    main()
