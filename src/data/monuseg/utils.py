from itertools import product

import json
import numpy as np
from scipy._lib.doccer import indentcount_lines


def extract_patches(image, patch_size=(100, 100)):
    h, w, _ = image.shape
    patches = list()
    for k, l in product(range(h // patch_size[0]), range(w // patch_size[1])):
        patches.append(image[k * patch_size[0]:(1 + k) * patch_size[0],
                             l * patch_size[1]:(1 + l) * patch_size[1], :])

    return np.stack(patches, axis=0)


def get_split(
    split,
    path_indices="/home/valentin/python_wkspce/2d_bispectrum_cnn/data/indices/monuseg.json"
):
    with open(path_indices, "r") as f:
        indices_list = json.load(f)

    return (
        indices_list[split]["train"],
        indices_list[split]["val"],
        indices_list[split]["test"],
    )
