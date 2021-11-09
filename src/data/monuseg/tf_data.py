from pathlib import Path
from random import shuffle, choice

import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
import numpy as np
from scipy.ndimage import rotate, median_filter

info_tissue_path = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/MoNuSeg2018_info.csv"


def get_dataset(
    id_list=None,
    data_path="/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training",
    is_test=False,
    instance=False,
):
    data_path = Path(data_path).resolve()

    image_dir = data_path / "Images_normalized"
    image_ids = [f.stem for f in Path(image_dir).rglob("*.tiff")]
    if id_list is not None:
        image_ids = [i for i in image_ids if i in id_list]

    if instance:

        def _load_image(image_id):
            return load_image_instance(image_id, data_path=data_path)
    else:

        def _load_image(image_id):
            return load_image(
                image_id,
                data_path=data_path,
                is_test=is_test,
            )

    return tf.data.Dataset.from_tensor_slices(image_ids).map(_load_image)


@tf.function
def to_sparse(mask):
    foreground = mask[..., 0]
    contour = mask[..., 1]
    output = foreground + 2.0 * contour
    output = tf.where(output > 2.0, x=2.0, y=output)
    return tf.expand_dims(output, -1)


# def to_sparse(mask):
#     w, h, _ = mask.shape
#     [mask] = tf.py_function(to_sparse_np, [mask], [tf.float32])
#     mask.set_shape((w, h, 1))
#     return mask

# def to_sparse_np(mask):
#     foreground = mask[..., 0]
#     contour = mask[..., 1]
#     output = foreground + 2.0 * contour
#     output = median_filter(output, size=(2, 2))
#     output[output == 1.5] = 1.0
#     output[output > 2.0] = 2.0
#     return output[..., np.newaxis]


def get_split(csv_path=info_tissue_path):
    info_df = pd.read_csv(csv_path)
    tissue_list = ["Breast", "Liver", "Kidney", "Prostate"]
    ids_train = list()
    ids_val = list()
    ids_test = list()
    for tissue in tissue_list:
        ids = info_df[info_df["organ"] == tissue]["patient_id"].values
        shuffle(ids)
        ids_train.extend(ids[:3])
        ids_val.extend([ids[3]])
        ids_test.extend(ids[4:5])
    return ids_train, ids_val, ids_test


def load_image(image_id, data_path="", is_test=False):
    # if type(image_id) is not str:
    #     image_id = tf.strings.as_string(image_id)
    image_dir = str(data_path / "Images_normalized")
    segmentation_dir = str(data_path / "Masks/color")
    contour_dir = str(data_path / "Masks/contours")

    path_image = tf.strings.join([image_dir, "/", image_id, ".tiff"])
    path_segmentation = tf.strings.join(
        [segmentation_dir, "/", image_id, ".tiff"])
    path_contour = tf.strings.join([contour_dir, "/", image_id, ".tiff"])

    image = load_image_tif(path_image, normalizing_factor=255.0)
    contour = load_binary_mask(path_contour)
    if is_test == False:
        segmentation = load_binary_mask(path_segmentation)
        segmentation = tf.where(
            (segmentation != 0) & (contour == 0),
            x=1.0,
            y=0.0,
        )
        background = tf.where(
            (segmentation != 0) | (contour != 0),
            x=0.0,
            y=1.0,
        )
        segmentation = tf.concat([segmentation, contour, background], axis=-1)
    else:
        segmentation = load_image_tif(path_segmentation,
                                      normalizing_factor=255.0)

    return image, segmentation


def load_image_instance(image_id, data_path=""):
    image_dir = str(data_path / "Images_normalized")
    segmentation_dir = str(data_path / "MasksV2_instance/binary")

    path_image = tf.strings.join([image_dir, "/", image_id, ".tiff"])
    path_segmentation = tf.strings.join(
        [segmentation_dir, "/", image_id, ".npy"])

    image = load_image_tif(path_image, normalizing_factor=255.0)
    segmentation = tf_load_npy_file(path_segmentation)

    return image, segmentation


def tf_load_npy_file(path):
    [out] = tf.py_function(load_npy_file, [path], [tf.int64])
    return out


def load_npy_file(item):
    data = np.load(item.numpy().decode("utf-8"))
    data = data[..., np.newaxis]
    return data.astype(np.int64)


def load_image_tif(path, normalizing_factor=1.0):
    image = tf.io.read_file(path)
    image = tfio.experimental.image.decode_tiff(image)[:, :, :3]
    return tf.cast(image, tf.float32) / normalizing_factor


def load_binary_mask(path):
    mask = tf.io.read_file(path)
    mask = tfio.experimental.image.decode_tiff(mask)[:, :, :3]
    mask = tf.image.rgb_to_grayscale(mask)
    return tf.where(mask > 0, x=1.0, y=0.0)


def tf_random_flip(image, segmentation):
    im_shape = image.shape
    seg_shape = segmentation.shape
    [image, segmentation] = tf.py_function(random_flip, [image, segmentation],
                                           [tf.float32, tf.float32])
    image.set_shape(im_shape)
    segmentation.set_shape(seg_shape)
    segmentation = tf.where(segmentation > 0.5, x=1.0, y=0.0)
    return image, segmentation


def tf_random_rotate(image, segmentation):
    im_shape = image.shape
    seg_shape = segmentation.shape
    [image,
     segmentation] = tf.py_function(random_rotate, [image, segmentation],
                                    [tf.float32, tf.float32])
    image.set_shape(im_shape)
    segmentation.set_shape(seg_shape)
    segmentation = tf.where(segmentation > 0.5, x=1.0, y=0.0)
    return image, segmentation


def random_flip(*images):
    random_flip_1 = np.random.rand() > 0.5
    random_flip_2 = np.random.rand() > 0.5
    output_images = list()
    for image in images:
        output_image = image
        if random_flip_1:
            output_image = image[::-1, :]
        if random_flip_2:
            output_image = output_image[:, ::-1]
        output_images.append(output_image)
    return output_images


def random_rotate(*images):
    output_images = list()
    angle = np.random.uniform(-180, 180)
    for image in images:
        output_images.append(rotate(image, angle, reshape=False, order=1))
    return output_images


def tf_random_crop(
        image,
        segmentation,
        size=(256, 256),
        rotation_angle=None,
        filter_segmentation=False,
):
    _random_crop = lambda x, y: random_crop(
        x,
        y,
        size=size,
        rotation_angle=rotation_angle,
        filter_segmentation=filter_segmentation,
    )
    [image, segmentation] = tf.py_function(_random_crop, [image, segmentation],
                                           (tf.float32, tf.float32))
    image.set_shape(size + (3, ))
    segmentation.set_shape(size + (3, ))
    return image, segmentation


def random_crop(image,
                segmentation,
                size=(256, 256),
                rotation_angle=None,
                filter_segmentation=False):
    image_height, image_width, _ = image.shape
    radius = np.sqrt(size[0]**2 + size[1]**2) / 2
    if type(rotation_angle) == float or type(rotation_angle) == int:
        angle = np.random.uniform(-rotation_angle, rotation_angle)
        dx = int((2 * radius - size[0]) // 2)
        dy = int((2 * radius - size[1]) // 2)
    elif rotation_angle == "right-angle":
        angle = choice([0, 90, 180, 270])
        dx, dy = 0, 0
    else:
        angle = None
        dx, dy = 0, 0
    offset_height = np.random.randint(dx, high=image_height - size[0] - dx)
    offset_width = np.random.randint(dy, high=image_width - size[1] - dy)
    if angle is not None:
        image_cropped = image[offset_height - dx:offset_height + dx + size[0],
                              offset_width - dy:offset_width + dy + size[1]]
        seg_cropped = segmentation[offset_height - dx:offset_height + dx +
                                   size[0], offset_width - dy:offset_width +
                                   dy + size[1]]

        image_rotated = rotate(image_cropped, angle, reshape=False, order=1)
        seg_rotated = rotate(seg_cropped, angle, reshape=False, order=1)
        if filter_segmentation:
            seg_rotated = median_filter(seg_rotated, size=(2, 2, 1))
        seg_rotated = np.where(seg_rotated > 0.5, 1.0, 0.0)
        return (
            image_rotated[dx:dx + size[0], dy:dy + size[1]],
            seg_rotated[dx:dx + size[0], dy:dy + size[1]],
        )

    else:
        return (image[offset_height:offset_height + size[0],
                      offset_width:offset_width + size[1]],
                segmentation[offset_height:offset_height + size[0],
                             offset_width:offset_width + size[1]])
