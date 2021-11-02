from pathlib import Path
from multiprocessing import Pool
from functools import partial

import click
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage.segmentation import mark_boundaries

default_seg_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Masks/color"
default_output_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Masks/contours"


@click.command()
@click.argument("seg_folder",
                type=click.Path(exists=True),
                default=default_seg_folder)
@click.argument("output_folder",
                type=click.Path(exists=True),
                default=default_output_folder)
@click.option("--mode", type=click.STRING, default="thick")
def main(seg_folder, output_folder, mode):
    output_folder = Path(output_folder)
    files = [f for f in tqdm(Path(seg_folder).rglob("*.tiff"))]

    fun = partial(compute_file, output_folder=output_folder, mode=mode)

    with Pool(10) as p:
        out = p.map(fun, files)


def compute_file(f, mode="thick", output_folder=""):
    segmentation = np.array(Image.open(f))
    seg_border = get_border(segmentation, mode=mode)
    Image.fromarray(seg_border).save(str(output_folder / f.name))
    return 1


def get_border(segmentation, mode="thick"):
    colors = np.unique(segmentation.reshape(-1, segmentation.shape[2]), axis=0)
    seg_border = np.zeros_like(segmentation, dtype=np.float64)
    for c_index in tqdm(range(colors.shape[0])):
        color = colors[c_index, :]
        mask = np.all(segmentation == color, axis=-1)
        seg_border = mark_boundaries(
            seg_border,
            mask,
            color=color / 255.0,
            mode=mode,
        )
    return (seg_border * 255.0).astype(segmentation.dtype)


if __name__ == '__main__':
    main()