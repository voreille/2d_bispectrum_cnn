from itertools import product

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from src.models.layers import (conv2d_complex, BCHConv2DComplex, tri,
                               ECHConv2D)


def tri(x):
    return np.where(np.abs(x) <= 1, np.where(x < 0, x + 1, 1 - x), 0)


def get_atoms(kernel_size=3, degrees=5):
    radius = (kernel_size - 1) // 2
    x_grid = np.arange(-radius, radius + 1, 1)
    x, y = np.meshgrid(x_grid, x_grid)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    atoms = np.zeros(
        (
            kernel_size,
            kernel_size,
            degrees + 1,
            kernel_size // 2 + 1,
        ),
        dtype=np.csingle,
    )
    for i, n in product(range(kernel_size // 2 + 1), range(degrees + 1)):
        # atoms[:, :, 0, 0, n, i] = (np.exp(-0.5 * ((r - i) / sigma)**2) *
        #                            np.exp(theta * n * 1j))
        atoms[:, :, n, i] = (tri(r - i) * np.exp(theta * n * (1j)))
    atoms[kernel_size // 2, kernel_size // 2, :, :] = 0
    norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1)))
    norm[norm == 0] = 1
    # atoms = atoms / norm

    return atoms / norm


max_degree = 5
kernel_size = 9
shs = get_atoms(kernel_size=kernel_size, degrees=max_degree)
shs = np.sum(shs, axis=-1)

factor = np.zeros((shs.shape[-1]))
# factor[1] = 1
factor[2] = 1
# factor[3] = 1
f1_proto = factor * shs
# phi = np.random.uniform(low=-np.pi, high=np.pi)
phi = np.pi * 0.25
rotation_factor = np.exp(1j * np.arange(shs.shape[-1]) * phi)
f2_proto = rotation_factor * f1_proto
f1 = np.real(np.sum(f1_proto, axis=-1))
f2 = np.real(np.sum(f2_proto, axis=-1))
f1_rotated = rotate(f1, phi / np.pi * 180, reshape=False)

# layer = ECHConv2D(1,
#                   9,
#                   max_degree=max_degree,
#                   initializer=tf.keras.initializers.Constant(1.0))
layer = BCHConv2DComplex(1,
                         9,
                         max_degree=max_degree,
                         initializer=tf.keras.initializers.Constant(1.0))

x = np.stack([f1, f2, f1_rotated], axis=0)
x = x[..., np.newaxis]
x = tf.constant(x)

y = layer(x).numpy()

central_responses = y[:, 0, 0, :]

print(np.abs(central_responses[0, ...]))
