from itertools import product

import numpy as np


def extract_patches(image, patch_size=(100, 100)):
    h, w, _ = image.shape
    patches = list()
    for k, l in product(range(h // patch_size[0]), range(w // patch_size[1])):
        patches.append(image[k * patch_size[0]:(1 + k) * patch_size[0],
                             l * patch_size[1]:(1 + l) * patch_size[1], :])

    return np.stack(patches, axis=0)
