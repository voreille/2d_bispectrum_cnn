import os
import datetime
import json
from pathlib import Path

import tensorflow as tf
import numpy as np
import click
from tensorflow.python.keras.metrics import AUC
import yaml
import pandas as pd

from src.models.monuseg.light_models import get_model
from src.models.loss import dice_coe_loss
from src.data.monuseg.tf_data import (get_dataset, tf_random_crop,
                                      tf_random_flip, get_split)
from src.models.monuseg.evaluation import post_processing
from src.models.monuseg.metrics import f_score, aggregated_jaccard_index
from src.models.callbacks import EarlyStopping

image_dir = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Images_normalized"
path_indices = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/indices/monuseg.json"
default_config_path = "/home/valentin/python_wkspce/2d_bispectrum_cnn/src/models/monuseg/configs/unetlight_default.yaml"

DEBUG = False

w_fg = 1.9
w_border = 5.0
w_bg = 0.44


def loss(y_true, y_pred):
    l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    w = (y_true[..., 0] * w_fg + y_true[..., 1] * w_border +
         y_true[..., 2] * w_bg)
    return (tf.reduce_mean(w * l, axis=(1, 2)) +
            dice_coe_loss(y_true[..., 0], y_pred[..., 0]) +
            dice_coe_loss(y_true[..., 1], y_pred[..., 1]))


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
@click.option("--config",
              type=click.Path(exists=True),
              default=default_config_path)
@click.option("--gpu_id", type=click.STRING, default='0')
@click.option("--n_rep", type=click.INT, default=10)
@click.option("--output_path",
              type=click.Path(),
              default="reports/results/MoNuSeg")
def main(config, gpu_id, n_rep, output_path):

    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    model_name = params["model_name"]
    rotation = params["rotation"]
    n_harmonics = params["n_harmonics"]
    cosine_decay = params["cosine_decay"]
    n_train = params["n_train"]

    result = pd.DataFrame()
    output_name = (model_name + f"__rotation_{rotation}__" +
                   f"nh_{n_harmonics}__" + f"n_train_{n_train}__" +
                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    for k in range(n_rep):
        result = result.append(
            train_one_split(
                split=k,
                gpu_id=gpu_id,
                model_name=model_name,
                rotation=rotation,
                n_harmonics=n_harmonics,
                cosine_decay=cosine_decay,
                n_train=n_train,
            ),
            ignore_index=True,
        )

    result.to_csv(Path(output_path) / (output_name + ".csv"))


def train_one_split(
    split=0,
    gpu_id='0',
    model_name=True,
    rotation="",
    n_harmonics=4,
    cosine_decay=False,
    n_train=-1,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    with open(path_indices, "r") as f:
        indices_list = json.load(f)

    ds_train = get_dataset(id_list=indices_list[split]["train"])
    ds_train = ds_train.cache().repeat(15)
    rotation_angle = None
    if rotation:
        rotation_angle = "right-angle"
    ds_train = ds_train.map(lambda image, seg: tf_random_crop(
        image, seg, rotation_angle=rotation_angle)).batch(4)

    ds_val = get_dataset(id_list=indices_list[split]["val"])
    ds_val = ds_val.cache().batch(1)

    ds_val_instance = get_dataset(id_list=indices_list[split]["val"],
                                  instance=True)
    ds_val_instance = ds_val_instance.cache().batch(1)

    ds_test = get_dataset(id_list=indices_list[split]["test"], instance=True)
    ds_test = ds_test.cache().batch(1)

    dir_name = (model_name + f"__rotation_{rotation}__" + f"split__{split}__" +
                f"nh_{n_harmonics}__" + f"n_train_{n_train}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = list()
    if not DEBUG:
        val_images = list()
        val_segs = list()
        for image, seg in ds_val:
            val_images.append(image)
            val_segs.append(seg)
        val_images = tf.concat(val_images, 0)
        val_segs = tf.concat(val_segs, 0)

        log_dir = (
            "/home/valentin/python_wkspce/2d_bispectrum_cnn/logs/fit_monuseg/"
            + dir_name)

        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')
        file_writer_eval_val = tf.summary.create_file_writer(
            log_dir + '/eval/validation')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            val_pred = model.predict(val_images)

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Validation images", val_images, step=epoch)
                tf.summary.image("Predictions", val_pred, step=epoch)
                tf.summary.image("GTs", val_segs, step=epoch)

        def log_eval(epoch, logs):
            aij, fscore = eval(ds=ds_val_instance, model=model)
            with file_writer_eval_val.as_default():
                tf.summary.scalar("Aggregated Jaccard Index", aij, step=epoch)
                tf.summary.scalar("F-score", fscore, step=epoch)
            logs["val_aij"] = aij
            logs["val_fscore"] = fscore
            print(f"val_aij: {aij} - val_fscore: {fscore}")

        callbacks.extend([
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_prediction),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_eval),
            EarlyStopping(
                minimal_num_of_epochs=10,
                monitor='val_fscore',
                patience=5,
                verbose=0,
                mode='max',
                restore_best_weights=True,
            ),
        ])

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
        epochs=150,
        validation_data=ds_val,
        callbacks=callbacks,
    )

    model.save_weights("models/" + dir_name + "/final")
    test_aij, test_fscore = eval(ds=ds_test, model=model)
    val_aij, val_fscore = eval(ds=ds_val_instance, model=model)
    return {
        "split": split,
        "test_aij": test_aij,
        "test_fscore": test_fscore,
        "val_aij": val_aij,
        "val_fscore": val_fscore,
    }


if __name__ == '__main__':
    main()
