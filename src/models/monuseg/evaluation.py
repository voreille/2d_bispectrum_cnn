from pathlib import Path

import numpy as np
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


def evaluate_equivariance(
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
    for image_id in image_ids:
        path_image = tf.strings.join([image_dir, "/", image_id, ".tiff"])
        image = load_image_tif(path_image, normalizing_factor=255.0).numpy()
        images = [image]
        for rot in [90, 180, 270]:
            images.append(rotate(image, rot, reshape=False))

        images = np.stack(images, axis=0)
        preds = model.predict(images, batch_size=1)
        pp_preds = post_processing(preds) != 0

        results_dict = {}
        for i, rot in enumerate([-90, -180, -270]):
            pred_rotated = rotate(preds[i + 1, ...], rot)
            pp_pred_rotated = rotate(pp_preds[i + 1, ...], rot)

            results_dict.update({
                f"RMSE_{-rot}deg_core":
                rmse(preds[0, :, :, 0], pred_rotated[..., 0]),
                f"RMSE_{-rot}deg_border":
                rmse(preds[0, :, :, 1], pred_rotated[..., 1]),
                f"RMSE_{-rot}deg_background":
                rmse(preds[0, :, :, 2], pred_rotated[..., 2]),
                f"dice_{-rot}deg":
                dice(pp_preds[0, ...], pp_pred_rotated),
            })

        results = results.append(
            {
                "image_id": image_id,
                **results_dict
            },
            ignore_index=True,
        )
    return results


def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2)**2))


def dice(x1, x2):
    return 2 * np.sum(np.logical_and(x1, x2)) / (np.sum(x1) + np.sum(x2))
