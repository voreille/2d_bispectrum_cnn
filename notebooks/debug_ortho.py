from itertools import product

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from src.models.layers import (conv2d_complex, BCHConv2DComplex, tri,
                               ECHConv2D)

layer = ECHConv2D(1, 9, initializer=tf.keras.initializers.Constant(1.0))

atoms0, atoms = layer.conv_ch.atoms

# %%
phi = np.pi * 0.5
rotation_factor = np.array([np.exp(1j * k * phi) for k in range(4)])

x = np.real(
    np.sum(np.sum(np.squeeze(atoms), axis=-1) *
           np.array([1, np.exp(1j * np.pi / 2), 1, 1]),
           axis=-1))
x_rotated = np.real(
    np.sum(np.sum(np.squeeze(atoms), axis=-1) *
           np.array([1, np.exp(1j * np.pi / 2), 1, 1]) * rotation_factor,
           axis=-1))

x_tot = np.stack([x, x_rotated], axis=0)
x_tot = x_tot[..., np.newaxis]
y = layer(x_tot)
