import tensorflow as tf

from src.models.layers import BCHConv2D

x = tf.random.uniform((32, 128, 128, 4))

layer = BCHConv2D(3, 4, 3)
y = layer(x)
print(f"y shape {y.shape}")