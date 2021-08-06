import os
from random import shuffle
import datetime

import tensorflow as tf
from tensorflow.keras import callbacks

from src.models.models import Unet, unet_model
from src.models.loss import dice_coe_loss, dice_coe_metric
from src.data.drive import (get_dataset, tf_random_crop, tf_random_rotate)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    image_ids = [i for i in range(21, 41)]
    shuffle(image_ids)
    image_ids_val = image_ids[:5]
    image_ids_train = image_ids[5:]

    ds_train = get_dataset(id_list=image_ids_train)
    ds_train = ds_train.cache().repeat(100).map(tf_random_rotate).map(
        tf_random_crop).map(lambda x, y, z: (x, z)).batch(16)

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

    log_dir = ("/home/valentin/python_wkspce/2d_bispectrum_cnn/logs/fit/" +
               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model = Unet(output_channels=1)
    model.compile(
        loss=[dice_coe_loss],
        optimizer=optimizer,
        metrics=[dice_coe_metric],
        # run_eagerly=True,
    )
    model.fit(
        x=ds_train,
        epochs=1000,
        validation_data=ds_val,
        callbacks=[tensorboard_callback, pred_callback],
    )


if __name__ == '__main__':
    main()
