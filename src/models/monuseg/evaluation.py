from operator import pos
from pathlib import Path
from itertools import combinations

import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from skimage.measure import label
from scipy.ndimage import watershed_ift, rotate

from src.data.monuseg.tf_data import load_image_tif


def post_processing(y_pred):
    if type(y_pred) != np.ndarray:
        y_pred_quantized = (y_pred.numpy() > 0.5).astype(np.uint8)
    else:
        y_pred_quantized = (y_pred > 0.5).astype(np.uint8)
    # y_pred_quantized = np.zeros_like(y_pred, dtype=np.uint8)
    # y_pred_quantized[..., 1] = (y_pred[..., 1] > 0.5).astype(np.uint8)
    # y_pred_quantized[..., 0] = (y_pred[..., 0] > 0.5).astype(np.uint8)
    # y_pred_quantized[..., 2] = (y_pred[..., 2] > 0.5).astype(np.uint8)
    batch_size = y_pred.shape[0]
    output = list()
    for s in range(batch_size):
        markers = label(y_pred_quantized[s, :, :, 0])
        markers[y_pred_quantized[s, :, :, 2] != 0] = -1
        out = watershed_ift((y_pred_quantized[s, :, :, 1]).astype(np.uint8),
                            markers)
        out[out == -1] = 0
        output.append(out)
    return np.stack(output, axis=0)


def rotate_batch(images, angle=90):
    output = np.zeros_like(images)
    for n in range(images.shape[0]):
        output[n, ...] = rotate(images[n, ...], angle=angle, reshape=False)
    return output


def disk(size):
    output = np.zeros((size, size))
    x = np.linspace(-size // 2, size // 2, num=size)
    x, y = np.meshgrid(x, x, indexing="xy")
    radius = np.sqrt(x**2 + y**2)
    output[radius <= size // 2 + 1] = 1
    return output


def evaluate_equivariance(
    model,
    id_list=None,
    data_path="/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training",
    n_angles=4,
    mask=True,
):

    delta_deg = 360 / n_angles
    rotations = [k * delta_deg for k in range(n_angles)]

    data_path = Path(data_path).resolve()
    image_dir = str(data_path / "Images_normalized")
    image_ids = [f.stem for f in Path(image_dir).rglob("*.tiff")]
    if id_list is not None:
        image_ids = [i for i in image_ids if i in id_list]

    results = pd.DataFrame()
    for image_id in tqdm(image_ids):
        path_image = tf.strings.join([image_dir, "/", image_id, ".tiff"])
        image = load_image_tif(path_image, normalizing_factor=255.0).numpy()
        preds = list()
        pp_preds = list()
        m = None
        for r in rotations:
            pred = rotate_batch(
                model(rotate_batch(image[np.newaxis, ...], r)),
                -r,
            )
            if m is None and mask:
                m = disk(pred.shape[1])

            pp_pred = post_processing(pred)
            preds.append(pred)
            pp_preds.append(pp_pred != 0)

        for r1, r2 in combinations(range(n_angles), 2):
            pred_1 = preds[r1]
            pred_2 = preds[r2]

            pp_pred_1 = pp_preds[r1]
            pp_pred_2 = pp_preds[r2]

            results_dict = {
                "rot_pair": (rotations[r1], rotations[r2]),
                "RMSE_core":
                rmse(pred_1[0, ..., 0], pred_2[0, ..., 0], mask=m),
                "RMSE_border":
                rmse(pred_1[0, ..., 1], pred_2[0, ..., 1], mask=m),
                "RMSE_background":
                rmse(pred_1[0, ..., 2], pred_2[0, ..., 2], mask=m),
                "dice":
                dice(pp_pred_1[0, ...], pp_pred_2[0, ...], mask=m),
            }

            results = results.append(
                {
                    "image_id": image_id,
                    **results_dict
                },
                ignore_index=True,
            )
    return results


def evaluate_equivariance_old(
    model,
    id_list=None,
    data_path="/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training",
):
    data_path = Path(data_path).resolve()
    image_dir = str(data_path / "Images_normalized")
    image_ids = [f.stem for f in Path(image_dir).rglob("*.tiff")]
    if id_list is not None:
        image_ids = [i for i in image_ids if i in id_list]
    results = pd.DataFrame()
    for image_id in tqdm(image_ids):
        path_image = tf.strings.join([image_dir, "/", image_id, ".tiff"])
        image = load_image_tif(path_image, normalizing_factor=255.0).numpy()
        images = [image]
        for rot in [90, 180, 270]:
            images.append(rotate(image, rot, reshape=False))

        images = np.stack(images, axis=0)
        preds = model.predict(images, batch_size=4)
        pp_preds = post_processing(preds) != 0

        results_dict = {}
        for i, rot in enumerate([-90, -180, -270]):
            preds[i + 1, ...] = rotate(preds[i + 1, ...], rot)
            pp_preds[i + 1, ...] = rotate(pp_preds[i + 1, ...], rot)

        rotations = [0, 90, 180, 270]
        for i, k in combinations([0, 1, 2, 3], 2):
            results_dict.update({
                "rot_pair": (rotations[i], rotations[k]),
                "RMSE_core":
                rmse(preds[i, :, :, 0], preds[k, :, :, 0]),
                "RMSE_border":
                rmse(preds[i, :, :, 1], preds[k, :, :, 1]),
                "RMSE_background":
                rmse(preds[i, :, :, 2], preds[k, :, :, 2]),
                "dice":
                dice(pp_preds[i, ...], pp_preds[k, ...]),
            })

            results = results.append(
                {
                    "image_id": image_id,
                    **results_dict
                },
                ignore_index=True,
            )
    return results


def rmse(x1, x2, mask=None):
    square_diff = (x1 - x2)**2
    if mask is not None:
        return np.sqrt(np.mean(square_diff[mask != 0]))
    else:
        return np.sqrt(np.mean(square_diff))


def dice(x1, x2, mask=None):
    if mask is not None:
        return 2 * np.sum((x1 != 0) & (x2 != 0) & (mask != 0)) / (np.sum(
            (x1 != 0) & (mask != 0)) + np.sum((x2 != 0) & (mask != 0)))
    else:
        return 2 * np.sum((x1 != 0) & (x2 != 0)) / (np.sum((x1 != 0)) + np.sum(
            (x2 != 0)))
