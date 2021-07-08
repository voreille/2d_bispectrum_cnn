import tensorflow as tf
import numpy as np

from src.models.layers import BCHConv2D, BCHConv2DComplex

x = tf.random.uniform((32, 128, 128, 4))

degrees = 4
layer = BCHConv2D(
    1,
    3,
    degrees=degrees,
    initializer=tf.keras.initializers.Constant(value=1),
)
y = layer(x)


angles = [np.mean(np.abs(y[..., k])) for k in range(y.shape[-1])]
print(angles)
print(f"y shape {y.shape}")