import os
from random import shuffle
import datetime
import json

import tensorflow as tf
import pandas as pd
import click
from tensorflow.python.keras.metrics import AUC

from src.models.models import get_model
from src.models.loss import loss_with_fl, dice_coe_metric, dice_coe_loss
from src.data.drive import (get_dataset, tf_random_crop, tf_random_rotate,
                            tf_random_flip)
from src.models.callbacks import EarlyStopping

default_indices = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/indices/drive.json"


def augment(image):
    image = tf.image.random_hue(image, 0.05)
    return tf.image.random_saturation(image, 0.8, 1.2)


@click.command()
@click.option('--model_name', type=click.STRING, default="UnetLight")
@click.option('--rotation/--no-rotation', default=True)
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('--focal_loss/--no-focal_loss', default=True)
@click.option('--cosine-decay/--no-cosine-decay', default=False)
@click.option('--cuda_core_id', type=click.STRING, default="1")
@click.option('--n_harmonics', type=click.INT, default=4)
@click.option('--n_train', type=click.INT, default=-1)
@click.option('--path-indices',
              type=click.Path(exists=True),
              default=default_indices)
def main(model_name, rotation, cuda_core_id, augmentation, focal_loss,
         cosine_decay, n_harmonics, path_indices, n_train):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_core_id

    with open(path_indices, "r") as f:
        indices_list = json.load(f)

    results_df = pd.DataFrame()

    for k, ind_dict in enumerate(indices_list):
        if n_train < 0:
            ind_train = ind_dict["train"]
        else:
            ind_train = ind_dict["train"][:n_train]
        ds_train = get_dataset(id_list=ind_train)
        ds_train = ds_train.cache().repeat(10)
        if rotation:
            ds_train = ds_train.map(tf_random_rotate).map(tf_random_flip)
        if augmentation:
            ds_train = ds_train.map(lambda x, y, z: (augment(x), y, z))
        ds_train = ds_train.map(tf_random_crop).map(lambda x, y, z:
                                                    (x, z)).batch(2)

        f = lambda x: tf.image.resize_with_crop_or_pad(x, 592, 592)

        ds_val = get_dataset(id_list=ind_dict["val"]).map(
            lambda x, y, z: (f(x), f(z))).cache().batch(1)
        ds_test = get_dataset(id_list=ind_dict["test"]).map(
            lambda x, y, z: (f(x), f(z))).cache().batch(1)
        ds_train_for_eval = get_dataset(id_list=ind_train).map(
            lambda x, y, z: (f(x), f(z))).cache().batch(1)

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
                    f"nh_{n_harmonics}__" + f"n_train_{len(ind_train)}__" +
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

        if focal_loss:
            loss = loss_with_fl
        else:
            loss = dice_coe_loss

        model = get_model(model_name=model_name,
                          loss=loss,
                          metrics=[dice_coe_metric, AUC()],
                          n_harmonics=n_harmonics,
                          cosine_decay=cosine_decay)

        es_callback = EarlyStopping(
            minimal_num_of_epochs=1,
            monitor='val_dice_coe_metric',
            patience=1,
            verbose=0,
            mode='max',
            restore_best_weights=True,
        )

        history = model.fit(
            x=ds_train,
            epochs=500,
            validation_data=ds_val,
            callbacks=[tensorboard_callback, pred_callback, es_callback],
        )
        res_train = model.evaluate(x=ds_train_for_eval)
        res_test = model.evaluate(x=ds_test)
        res_val = model.evaluate(x=ds_val)
        results_df = results_df.append(
            {
                "train_loss": res_train[0],
                "train_dsc": res_train[1],
                "train_auc": res_train[2],
                "val_loss": res_val[0],
                "val_dsc": res_val[1],
                "val_auc": res_val[2],
                "test_loss": res_test[0],
                "test_dsc": res_test[1],
                "test_auc": res_test[2],
                "repetition": k,
                "epochs": len(history.history["loss"]),
                "steps_per_epoch": history.params["steps"],
            },
            ignore_index=True,
        )
    results_df.to_csv(f"/home/valentin/python_wkspce/2d_bispectrum_cnn"
                      f"/reports/results/{dir_name}.csv")


if __name__ == '__main__':
    main()
