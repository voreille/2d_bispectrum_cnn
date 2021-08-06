from pathlib import Path

import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO


def get_coco_dataset(
    split="train",
    data_path="/home/valentin/datasets/coco/data",
    filter_classes=None,
    contain_all_classes=False,
    coco=None,
):
    data_path = Path(data_path)
    images_dir = data_path / f"{split}2017"
    annotations_path = data_path / f"annotations/instances_{split}2017.json"
    if coco is None:
        coco = COCO(str(annotations_path))

    if filter_classes:
        category_ids = coco.getCatIds(catNms=filter_classes)
    else:
        category_ids = coco.getCatIds()

    if not contain_all_classes:
        image_ids = list()
        for i in category_ids:
            image_ids += coco.getImgIds(catIds=[i])
        image_ids = list(set(image_ids))
    else:
        image_ids = coco.getImgIds(catIds=category_ids)
    image_ids_ds = tf.data.Dataset.from_tensor_slices(image_ids)

    def load_image(image_id):

        image_metadata = coco.loadImgs([int(image_id.numpy())])[0]
        # im_shape = (image_metadata["height"], image_metadata["width"])

        image = tf.io.read_file(
            str((images_dir / image_metadata["file_name"]).resolve()))
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [128, 128])
        mask = get_mask(image_metadata, coco=coco, category_ids=category_ids)

        return image, mask

    def tf_load_image(image_id):
        [image, mask] = tf.py_function(load_image, [image_id],
                                       [tf.float32, tf.uint8])
        # mask.set_shape(image.shape)
        return image, mask

    return image_ids_ds.map(tf_load_image)


@tf.function
def get_mask(image_metadata, coco=None, category_ids=None):
    mask = tf.zeros((image_metadata['height'], image_metadata['width']),
                    dtype=tf.uint8)
    annotation_ids = coco.getAnnIds(imgIds=image_metadata['id'],
                                    catIds=category_ids,
                                    iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)

    for i in range(len(annotations)):
        pixel_value = category_ids.index(annotations[i]['category_id']) + 1
        mask = tf.maximum(
            tf.constant(coco.annToMask(annotations[i]), dtype=tf.uint8) *
            pixel_value, mask)

    return mask