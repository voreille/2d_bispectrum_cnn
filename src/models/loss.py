import tensorflow as tf
from tensorflow_addons.losses import sigmoid_focal_crossentropy


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


def dice_coe_loss(y_true, y_pred, loss_type='sorensen', smooth=1.):
    return 1 - dice_coe(
        y_true,
        y_pred,
        loss_type=loss_type,
        smooth=smooth,
    )


def dice_coe_metric(y_true, y_pred, loss_type='sorensen', smooth=1.):
    return dice_coe(y_true,
                    tf.cast(y_pred > 0.5, tf.float32),
                    loss_type=loss_type,
                    smooth=smooth)


def dice_coe(y_true, y_pred, loss_type='jaccard', smooth=1., axis=(1, 2)):
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    if loss_type == 'jaccard':
        union = tf.reduce_sum(
            tf.square(y_pred),
            axis=axis,
        ) + tf.reduce_sum(tf.square(y_true), axis=axis)

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred, axis=axis) + tf.reduce_sum(y_true,
                                                                 axis=axis)
    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)
