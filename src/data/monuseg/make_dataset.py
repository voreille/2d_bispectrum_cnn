from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from skimage import draw
import numpy as np
import click
from tqdm import tqdm

default_image_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Images"
default_annotation_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Annotations"
default_output_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/MasksV2_instance"


@click.command()
@click.option("--image_folder",
              type=click.Path(exists=True),
              default=default_image_folder)
@click.option("--annotation_folder",
              type=click.Path(exists=True),
              default=default_annotation_folder)
@click.option("--output_folder",
              type=click.Path(),
              default=default_output_folder)
@click.option('--instance/--semantic', default=True)
def main(image_folder, annotation_folder, output_folder, instance):
    output_folder = Path(output_folder)
    binary_folder = output_folder / "binary/"
    color_folder = output_folder / "color/"
    annotation_folder = Path(annotation_folder)
    output_folder.mkdir(parents=False, exist_ok=True)
    binary_folder.mkdir(exist_ok=True)
    color_folder.mkdir(exist_ok=True)

    files_path = list(Path(image_folder).rglob("*.tif"))
    for f in tqdm(files_path):
        annotations_file = annotation_folder / (f.stem + ".xml")
        binary_mask, color_mask = he_to_binary_mask(
            f,
            annotations_file=annotations_file,
            instance=instance,
        )
        np.save(binary_folder / f"{f.stem}.npy", binary_mask)
        np.save(color_folder / f"{f.stem}.npy", color_mask)


def poly2mask(vertex_row_coords, vertex_col_coords, shape, value):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords,
                                                    vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.int)
    mask[fill_row_coords, fill_col_coords] = value
    return mask


def he_to_binary_mask(
    filepath,
    annotations_file,
    instance=False,
):
    """
    Convert XML annotation file to a mask image. 
    """
    tree = ET.parse(annotations_file)
    xDoc = tree.getroot()
    regions = xDoc.iter('Region')  # get a list of all the region tags
    array_xy = []

    for i, region in enumerate(regions):  # Region = nuclei
        #Region = Regions.item(regioni)    # for each region tag

        #get a list of all the vertexes (which are in order)
        xy = []
        for vertexi, vertex in enumerate(
                region.iter('Vertex')):  #iterate through all verticies
            #get the x value of that vertex
            x = float(vertex.attrib['X'])
            y = float(vertex.attrib['Y'])

            #get the y value of that vertex

            xy.append([x, y])  # finally save them into the array
        array_xy.append(xy)
    array_xy = np.array(array_xy)
    im = Image.open(filepath)
    ncol, nrow = im.size
    binary_mask = np.zeros((nrow, ncol))
    color_mask = np.zeros((3, nrow, ncol))

    #mask_final = [];
    for i, r in enumerate(array_xy):  #for each region
        smaller_x = np.array(r)[:, 0]
        smaller_y = np.array(r)[:, 1]
        if instance:
            value = i + 1
        else:
            value = 1
        polygon = poly2mask(smaller_x, smaller_y, (nrow, ncol), value=value)
        binary_mask = binary_mask + np.where(
            (polygon > 0) &
            (binary_mask > 0), 0, polygon)  # Where overlap -> 0
        color_mask = color_mask + np.stack(
            (np.random.rand() * polygon, np.random.rand() * polygon,
             np.random.rand() * polygon))

    binary_mask = binary_mask.T
    binary_mask = binary_mask.astype(int)
    color_mask = color_mask.T
    return binary_mask, color_mask


if __name__ == '__main__':
    main()