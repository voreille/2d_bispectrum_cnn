from pathlib import Path
from numpy.lib.function_base import kaiser

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scipy.ndimage import rotate


def get_dataset(
    id_list=None,
    data_path="/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training",
):
    data_path = Path(data_path).resolve()

    image_dir = data_path / "Images_normalized"
    image_ids = [f.stem for f in Path(image_dir).rglob("*.tiff")]
    if id_list:
        image_ids = [i for i in image_ids if i in id_list]

    def _load_image(image_id):
        return load_image(image_id, data_path=data_path)

    return tf.data.Dataset.from_tensor_slices(image_ids).map(_load_image)


def load_image(image_id, data_path=""):
    # if type(image_id) is not str:
    #     image_id = tf.strings.as_string(image_id)
    image_dir = str(data_path / "Images_normalized")
    segmentation_dir = str(data_path / "Masks/binary")

    path_image = tf.strings.join([image_dir, "/", image_id, ".tiff"])
    path_segmentation = tf.strings.join(
        [segmentation_dir, "/", image_id, ".tiff"])

    image = load_image_tif(path_image)
    segmentation = load_binary_mask(path_segmentation)
    return image, segmentation


def load_image_tif(path):
    image = tf.io.read_file(path)
    image = tfio.experimental.image.decode_tiff(image)[:, :, :3]
    return tf.cast(image, tf.float32) / 255.0


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


def tf_random_crop(image, segmentation, size=(256, 256), rotation=False):
    _random_crop = lambda x, y: random_crop(x, y, size=size, rotation=rotation)
    [image, segmentation] = tf.py_function(_random_crop, [image, segmentation],
                                           (tf.float32, tf.float32))
    image.set_shape(size + (3, ))
    segmentation.set_shape(size + (1, ))
    return image, segmentation


def random_crop(image, segmentation, size=(256, 256), rotation=False):
    image_height, image_width, _ = image.shape
    radius = int(np.sqrt(size[0]**2 + size[1]**2) // 2)
    if rotation:
        angle = np.random.uniform(-180, 180)
        dx = radius - size[0]
        dy = radius - size[1]
    else:
        dx, dy = 0, 0
    offset_height = np.random.randint(dx, high=image_height - size[0] - dx)
    offset_width = np.random.randint(dy, high=image_width - size[1] - dy)
    if rotation:
        image_cropped = image[offset_height - dx:offset_height + dx + size[0],
                              offset_width - dy:offset_width + dy + size[1]]
        seg_cropped = segmentation[offset_height - dx:offset_height + dx +
                                   size[0], offset_width - dy:offset_width +
                                   dy + size[1]]

        image_rotated = rotate(image_cropped, angle, reshape=False, order=1)
        seg_rotated = rotate(seg_cropped, angle, reshape=False, order=1)
        seg_rotated = tf.where(seg_rotated > 0.5, x=1.0, y=0.0)
        return (
            image_rotated[dx:size[0], dy:size[1]],
            seg_rotated[dx:size[0], dy:size[1]],
        )

    else:
        return (image[offset_height:offset_height + size[0],
                      offset_width:offset_width + size[1]],
                segmentation[offset_height:offset_height + size[0],
                             offset_width:offset_width + size[1]])

    return output_images
