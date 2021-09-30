from pathlib import Path
from numpy.lib.function_base import kaiser

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scipy.ndimage import rotate


def get_dataset(
    split="train",
    id_list=None,
    data_path="/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/drive",
):
    data_path = Path(data_path).resolve()

    image_dir = data_path / f"{split}/images"
    image_ids = [f.name.split("_")[0] for f in Path(image_dir).rglob("*.tif")]
    if id_list:
        id_list = [str(i) for i in id_list]
        image_ids = [i for i in image_ids if i in id_list]

    def _load_image(image_id):
        return load_image(image_id, data_path=data_path, split=split)

    return tf.data.Dataset.from_tensor_slices(image_ids).map(_load_image)


def load_image(image_id, data_path="", split="train"):
    # if type(image_id) is not str:
    #     image_id = tf.strings.as_string(image_id)
    image_dir = str(data_path / f"{split}/images")
    mask_dir = str(data_path / f"{split}/mask")
    segmentation_dir = str(data_path / f"{split}/1st_manual")

    path_image = tf.strings.join([image_dir, "/", image_id, "_training.tif"])
    path_mask = tf.strings.join(
        [mask_dir, "/", image_id, "_training_mask.gif"])
    path_segmentation = tf.strings.join(
        [segmentation_dir, "/", image_id, "_manual1.gif"])

    image = load_image_tif(path_image)
    mask = load_binary_mask(path_mask)
    segmentation = load_binary_mask(path_segmentation)
    return image, mask, segmentation


def load_image_tif(path):
    image = tf.io.read_file(path)
    image = tfio.experimental.image.decode_tiff(image)[:, :, :3]
    return tf.cast(image, tf.float32) / 255.0


def load_binary_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.io.decode_image(mask)[0, ...]
    mask = tf.image.rgb_to_grayscale(mask)
    return tf.where(mask > 0, x=1.0, y=0.0)


def tf_random_flip(image, mask, segmentation):
    im_shape = image.shape
    mask_shape = mask.shape
    seg_shape = segmentation.shape
    [image, mask,
     segmentation] = tf.py_function(random_flip, [image, mask, segmentation],
                                    [tf.float32, tf.float32, tf.float32])
    image.set_shape(im_shape)
    mask.set_shape(mask_shape)
    segmentation.set_shape(seg_shape)
    mask = tf.where(mask > 0.5, x=1.0, y=0.0)
    segmentation = tf.where(segmentation > 0.5, x=1.0, y=0.0)
    return image, mask, segmentation


def tf_random_rotate(image, mask, segmentation):
    im_shape = image.shape
    mask_shape = mask.shape
    seg_shape = segmentation.shape
    [image, mask,
     segmentation] = tf.py_function(random_rotate, [image, mask, segmentation],
                                    [tf.float32, tf.float32, tf.float32])
    image.set_shape(im_shape)
    mask.set_shape(mask_shape)
    segmentation.set_shape(seg_shape)
    mask = tf.where(mask > 0.5, x=1.0, y=0.0)
    segmentation = tf.where(segmentation > 0.5, x=1.0, y=0.0)
    return image, mask, segmentation


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


def tf_random_crop(image, mask, segmentation, size=(256, 256)):
    _random_crop = lambda x, y, z: random_crop(x, y, z, size=size)
    [image, mask,
     segmentation] = tf.py_function(_random_crop, [image, mask, segmentation],
                                    (tf.float32, tf.float32, tf.float32))
    image.set_shape(size + (3, ))
    mask.set_shape(size + (1, ))
    segmentation.set_shape(size + (1, ))
    return image, mask, segmentation


def random_crop(*images, size=(256, 256)):
    image_height, image_width, _ = images[0].shape
    offset_height = np.random.randint(0, high=image_height - size[0])
    offset_width = np.random.randint(0, high=image_width - size[1])
    output_images = list()
    for image in images:
        output_images.append(image[offset_height:offset_height + size[0],
                                   offset_width:offset_width + size[1]])

    return output_images
