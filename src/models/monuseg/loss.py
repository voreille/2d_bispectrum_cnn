import tensorflow as tf
from tensorflow_addons.losses import sigmoid_focal_crossentropy


def loss_with_fl(y_true, y_pred):
    return tf.reduce_mean(
        sigmoid_focal_crossentropy(
            y_true,
            y_pred,
            from_logits=False,
            alpha=0.25,
            gamma=2.0,
        ),
        axis=(1, 2),
    )
