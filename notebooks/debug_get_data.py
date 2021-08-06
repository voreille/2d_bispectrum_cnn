from src.data.drive import get_dataset, tf_random_crop

image_ids = [i for i in range(21, 41)]

ds = get_dataset(id_list=image_ids).map(tf_random_crop)
img, mask, segmentation = next(ds.take(1).as_numpy_iterator())
# img, mask = next(ds.take(1).as_numpy_iterator())
print(img)