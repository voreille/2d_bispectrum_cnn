import os
import datetime

import tensorflow as tf
import numpy as np
import click
from tensorflow.python.keras.metrics import AUC

from src.models.monuseg.light_models import get_model
from src.models.loss import dice_coe_loss
from src.data.monuseg.tf_data import (get_dataset, tf_random_crop,
                                      tf_random_flip, get_split)
from src.models.monuseg.evaluation import post_processing
from src.models.monuseg.metrics import f_score, aggregated_jaccard_index

image_dir = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Images_normalized"

DEBUG = False

w_fg = 1.9
w_border = 5.0
w_bg = 0.44


def loss(y_true, y_pred):
    l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    w = (y_true[..., 0] * w_fg + y_true[..., 1] * w_border +
         y_true[..., 2] * w_bg)
    # w = tf.where(y_true[..., 0] == 1, x=w_fg, y=0)
    # w = tf.where(y_true[..., 1] == 1, x=w_border, y=w)
    # w = tf.where(y_true[..., 2] == 1, x=w_bg, y=w)
    return tf.reduce_mean(w * l, axis=(1, 2)) + dice_coe_loss(
        y_true[..., 0], y_pred[..., 0])


def eval(ds=None, model=None):
    aij_list = list()
    fscore_list = list()
    for x, y_true in ds.as_numpy_iterator():
        y_pred = post_processing(model(x))
        for s in range(y_pred.shape[0]):
            aij_list.append(
                aggregated_jaccard_index(y_true[s, :, :, 0], y_pred[s, :, :]))
            fscore_list.append(f_score(y_true[s, :, :, 0], y_pred[s, :, :]))

    return np.mean(aij_list), np.mean(fscore_list)


@click.command()
@click.option('--model_name', type=click.STRING, default="BispectUnetLight")
@click.option('--rotation/--no-rotation', default=True)
@click.option('--cosine-decay/--no-cosine-decay', default=False)
@click.option('--cuda_core_id', type=click.STRING, default="1")
@click.option('--n_harmonics', type=click.INT, default=4)
@click.option('--n_train', type=click.INT, default=-1)
def main(model_name, rotation, cuda_core_id, cosine_decay, n_harmonics,
         n_train):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_core_id

    image_ids_train, image_ids_val, image_ids_test = get_split()

    ds_train = get_dataset(id_list=image_ids_train)
    ds_train = ds_train.cache().repeat(15)
    rotation_angle = None
    if rotation:
        rotation_angle = "right-angle"
    ds_train = ds_train.map(lambda image, seg: tf_random_crop(
        image, seg, rotation_angle=rotation_angle)).batch(4)

    ds_val = get_dataset(id_list=image_ids_val)
    ds_val = ds_val.cache().batch(1)

    ds_test = get_dataset(id_list=image_ids_test, instance=True)
    ds_test = ds_test.cache().batch(1)

    ds_val_instance = get_dataset(id_list=image_ids_val, instance=True)
    ds_val_instance = ds_val_instance.cache().batch(1)

    callbacks = list()
    if not DEBUG:
        val_images = list()
        val_segs = list()
        for image, seg in ds_val:
            val_images.append(image)
            val_segs.append(seg)
        val_images = tf.concat(val_images, 0)
        val_segs = tf.concat(val_segs, 0)

        dir_name = (model_name + f"__rotation_{rotation}__" +
                    f"cdecay__{cosine_decay}__" + f"nh_{n_harmonics}__" +
                    f"n_train_{n_train}__" +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_dir = (
            "/home/valentin/python_wkspce/2d_bispectrum_cnn/logs/fit_monuseg/"
            + dir_name)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)
        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')
        file_writer_eval_val = tf.summary.create_file_writer(
            log_dir + '/eval/validation')
        file_writer_eval_test = tf.summary.create_file_writer(log_dir +
                                                              '/eval/test')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            val_pred = model.predict(val_images)

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Validation images", val_images, step=epoch)
                tf.summary.image("Predictions", val_pred, step=epoch)
                tf.summary.image("GTs", val_segs, step=epoch)

        def log_eval(epoch, logs):
            aij, fscore = eval(ds=ds_test, model=model)
            with file_writer_eval_test.as_default():
                tf.summary.scalar("aij", aij, step=epoch)
                tf.summary.scalar("fscore", fscore, step=epoch)

            aij, fscore = eval(ds=ds_val_instance, model=model)
            with file_writer_eval_val.as_default():
                tf.summary.scalar("aij", aij, step=epoch)
                tf.summary.scalar("fscore", fscore, step=epoch)

        pred_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_prediction)
        eval_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_eval)
        callbacks.extend([tensorboard_callback, pred_callback, eval_callback])

    model = get_model(model_name=model_name,
                      output_channels=3,
                      loss=loss,
                      metrics=[AUC()],
                      n_harmonics=n_harmonics,
                      cosine_decay=cosine_decay,
                      last_activation="softmax",
                      run_eagerly=False)
    model.fit(
        x=ds_train,
        epochs=100,
        validation_data=ds_val,
        callbacks=callbacks,
    )
    result = model.predict(ds_test)
    test_images = list()
    test_seg = list()
    for x, y_true in ds_test.as_numpy_iterator():
        test_images.append(x)
        test_seg.append(y_true)
    y_true = np.concatenate(test_seg, axis=0)
    x = np.concatenate(test_images, axis=0)

    np.savez("test.npy", y_pred=result, x=x, y_true=y_true)


if __name__ == '__main__':
    main()
