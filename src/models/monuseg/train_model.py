import os
import datetime
import json
from pathlib import Path

import tensorflow as tf
import numpy as np
import click
import yaml
import pandas as pd

from src.models.monuseg.models_fixed import get_model
from src.data.monuseg.tf_data import get_dataset, tf_random_crop
from src.models.monuseg.evaluation import post_processing
from src.models.monuseg.metrics import (aggregated_jaccard_index,
                                        confusion_terms)
from src.models.callbacks import EarlyStopping

project_dir = Path(__file__).resolve().parents[3]

image_dir = project_dir / "data/raw/MoNuSeg2018Training/Images_normalized"
path_indices = project_dir / "data/indices/monuseg.json"
default_config_path = project_dir / "src/models/monuseg/configs/unet_default_fixed.yaml"

DEBUG = False

w_fg = 1.9
w_border = 5.0
w_bg = 0.44

# def loss(y_true, y_pred):
#     l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#     w = (y_true[..., 0] * w_fg + y_true[..., 1] * w_border +
#          y_true[..., 2] * w_bg)
#     return (tf.reduce_mean(w * l, axis=(1, 2)) +
#             dice_coe_loss(y_true[..., 0], y_pred[..., 0]) +
#             dice_coe_loss(y_true[..., 1], y_pred[..., 1]))


def loss(y_true, y_pred, cropper=None):
    if cropper is not None:
        y_true = cropper(y_true)
    l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    w = (y_true[..., 0] * w_fg + y_true[..., 1] * w_border +
         y_true[..., 2] * w_bg)
    return tf.reduce_mean(w * l, axis=(1, 2))


def eval(ds=None, model=None, cropper=None):
    scores_dict = {
        "fscore": [],
        "aij": [],
        "recall": [],
        "precision": [],
    }
    for x, y_true in ds.as_numpy_iterator():
        if cropper is not None:
            y_true = cropper(y_true).numpy()
        y_pred = post_processing(model(x, training=False))
        for s in range(y_pred.shape[0]):
            scores_dict["aij"].append(
                aggregated_jaccard_index(y_true[s, :, :, 0], y_pred[s, :, :]))
            fp, fn, tp = confusion_terms(y_true[s, :, :, 0], y_pred[s, :, :])
            scores_dict["fscore"].append(tp / (tp + 0.5 * (fp + fn)))
            scores_dict["recall"].append(tp / (tp + fn))
            scores_dict["precision"].append(tp /
                                            (tp + fp) if tp + fp != 0 else 0)

    return {k: np.mean(i) for k, i in scores_dict.items()}


def write_config_to_file(params, model, output_path=""):
    with open(output_path, "w") as f:
        print(
            28 * "=" + " CONFIG FILE " + 28 * "=" + "\n",
            file=f,
        )
        for key, item in params.items():
            print(f"{key}: {item}", file=f)
        print(
            "\n" + 25 * "=" + " MODEL ARCHITECTURE " + 25 * "=" + "\n",
            file=f,
        )

        model.summary(print_fn=lambda s: print(s, file=f))


def config_gpu(memory_limit):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(gpus[0], [
                tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)
            ])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


@click.command()
@click.option("--config",
              type=click.Path(exists=True),
              default=default_config_path)
@click.option("--gpu-id", type=click.STRING, default='3')
@click.option("--n-rep", type=click.INT, default=10)
@click.option("--split", type=click.INT, default=0)
@click.option('--train-rep/--no-train-rep', default=True)
@click.option('--label', type=click.STRING, default="")
@click.option("--output_path", type=click.Path(), default="models/MoNuSeg")
@click.option("--n-harmonics", type=click.INT, default=-1)
@click.option("--batch-size", type=click.INT, default=-1)
@click.option("--radial-profile-type", type=click.FLOAT, default=None)
@click.option("--epochs", type=click.INT, default=200)
@click.option("--learning-rate", type=click.FLOAT, default=1e-3)
def main(config, gpu_id, n_rep, split, train_rep, output_path, label,
         n_harmonics, batch_size, radial_profile_type, epochs, learning_rate):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # config_gpu()
    output_path = Path(output_path)
    if DEBUG:
        train_rep = True
        epochs = 1

    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    if n_harmonics > 0:
        params["n_harmonics"] = n_harmonics
    if batch_size > 0:
        params["batch_size"] = batch_size

    if radial_profile_type is not None:
        params["radial_profile_type"] = radial_profile_type

    model_name = params["model_name"]
    rotation = params["rotation"]
    n_harmonics = params["n_harmonics"]
    n_train = params["n_train"]
    patch_size = params["patch_size"]

    result = pd.DataFrame()
    output_name = (model_name + label + f"__rotation_{rotation}__" +
                   f"nh_{n_harmonics}__" + f"n_train_{n_train}__" +
                   f"lr_{learning_rate}__" +
                   f"psize_{patch_size[0]}x{patch_size[1]}__" +
                   f"rtype_{params['radial_profile_type']}" +
                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    dir_path = output_path / output_name
    dir_path.mkdir()

    write_config = True
    if train_rep:
        print(f"STARTING {n_rep} repetitions")
        for k in range(n_rep):
            result = result.append(
                train_one_split(
                    split=k,
                    label=label,
                    params=params,
                    write_config=write_config,
                    dir_path=dir_path,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    n_patches_per_epoch=params["n_patches_per_epoch"],
                ),
                ignore_index=True,
            )
            write_config = False

    else:
        print(8 * "$#!" + "WATCH OUT: you are running only one rep" +
              8 * "$#!")
        result = result.append(
            train_one_split(
                split=split,
                params=params,
                label=label,
                write_config=write_config,
                dir_path=dir_path,
                epochs=epochs,
                learning_rate=learning_rate,
            ),
            ignore_index=True,
        )

    result.to_csv(dir_path / "metrics.csv")


def train_one_split(
    split=0,
    label="",
    params=None,
    write_config=False,
    dir_path=None,
    epochs=200,
    learning_rate=1e-3,
    n_patches_per_epoch=209,
):
    model_name = params["model_name"]
    rotation = params["rotation"]
    n_harmonics = params["n_harmonics"]
    n_train = params["n_train"]
    n_feature_maps = params["n_feature_maps"]
    cosine_decay = params["cosine_decay"]
    batch_size = params["batch_size"]
    patch_size = tuple(params["patch_size"])
    radial_profile_type = params["radial_profile_type"]
    cropper = tf.keras.layers.Cropping2D(cropping=(20, 20))

    with open(path_indices, "r") as f:
        indices_list = json.load(f)

    ds_train = get_dataset(id_list=indices_list[split]["train"])
    ds_train = ds_train.cache().repeat(n_patches_per_epoch //
                                       len(indices_list[split]["train"]))

    # ds_train = ds_train.cache().repeat(1)

    if rotation:
        rotation_angle = "right-angle"
    else:
        rotation_angle = None
    ds_train = ds_train.map(
        lambda image, seg: tf_random_crop(
            image,
            seg,
            rotation_angle=rotation_angle,
            size=patch_size,
            random_brightness=True,
        ),
        num_parallel_calls=16,
    ).batch(batch_size)

    ds_val = get_dataset(id_list=indices_list[split]["val"])
    ds_val = ds_val.cache().batch(1)

    ds_val_instance = get_dataset(id_list=indices_list[split]["val"],
                                  instance=True)
    ds_val_instance = ds_val_instance.cache().batch(1)

    ds_test = get_dataset(id_list=indices_list[split]["test"], instance=True)
    ds_test = ds_test.cache().batch(1)

    dir_name = (model_name + label + f"__rotation_{rotation}__" +
                f"split__{split}__" + f"nh_{n_harmonics}__" +
                f"n_train_{n_train}__" + f"lr_{learning_rate}__" +
                f"psize_{patch_size[0]}x{patch_size[1]}__" +
                f"rtype_{radial_profile_type}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = list()
    if not DEBUG:
        val_segs = list()
        for _, seg in ds_val.as_numpy_iterator():
            val_segs.append(seg)
        val_segs = np.concatenate(val_segs, axis=0)

        val_images = list()
        val_seg_instance = list()
        for image, seg in ds_val_instance.as_numpy_iterator():
            val_images.append(image)
            val_seg_instance.append(seg)
        val_images = np.concatenate(val_images, axis=0)
        val_seg_instance = np.concatenate(val_seg_instance, axis=0)
        val_seg_instance = cropper(val_seg_instance).numpy()

        log_dir = (
            "/home/valentin/python_wkspce/2d_bispectrum_cnn/logs/fit_monuseg_lafarge/"
            + dir_name)

        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')
        file_writer_eval_val = tf.summary.create_file_writer(
            log_dir + '/eval/validation')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            val_pred = model.predict(val_images, batch_size=1)

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Validation images", val_images, step=epoch)
                tf.summary.image("Predictions", val_pred, step=epoch)
                tf.summary.image("GTs", val_segs, step=epoch)

            fscore_list = list()
            val_pred = post_processing(val_pred)
            for k in range(val_pred.shape[0]):
                fp, fn, tp = confusion_terms(val_seg_instance[k, :, :, 0],
                                             val_pred[k, :, :])

                fscore_list.append(tp / (tp + 0.5 * (fp + fn)))

            val_fscore = np.mean(fscore_list)
            logs["val_fscore"] = val_fscore
            print(f"val_fscore: {val_fscore}")
            with file_writer_eval_val.as_default():
                tf.summary.scalar("F-score", val_fscore, step=epoch)

        callbacks.extend([
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_prediction),
            EarlyStopping(
                minimal_num_of_epochs=10,
                monitor='val_fscore',
                patience=15,
                verbose=0,
                mode='max',
                restore_best_weights=True,
            ),
        ])

    model = get_model(
        model_name=model_name,
        output_channels=3,
        loss=lambda y_true, y_pred: loss(y_true, y_pred, cropper=cropper),
        n_harmonics=n_harmonics,
        cosine_decay=cosine_decay,
        n_feature_maps=n_feature_maps,
        last_activation="softmax",
        radial_profile_type=radial_profile_type,
        run_eagerly=False,
        lr=learning_rate,
    )

    model.fit(
        x=ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=callbacks,
    )

    if write_config and not DEBUG:
        write_config_to_file(
            params,
            model,
            output_path=dir_path / "config.txt",
        )

    if not DEBUG:
        dir_to_save_weights = dir_path / "weights" / f"split_{split}" / "final"
        model.save_weights(dir_to_save_weights)
    test_scores = eval(ds=ds_test, model=model, cropper=cropper)
    val_scores = eval(ds=ds_val_instance, model=model, cropper=cropper)
    test_scores = {f"test_{k}": i for k, i in test_scores.items()}
    val_scores = {f"val_{k}": i for k, i in val_scores.items()}
    return {"split": split, **test_scores, **val_scores}


if __name__ == '__main__':
    main()
