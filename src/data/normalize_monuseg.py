from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm

default_image_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Images"
default_output_folder = "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/raw/MoNuSeg2018Training/Images_normalized"


@click.command()
@click.argument("image_folder",
                type=click.Path(exists=True),
                default=default_image_folder)
@click.argument("output_folder",
                type=click.Path(exists=True),
                default=default_output_folder)
@click.option('--eosin-hematoxylin/--no-eosin-hematoxylin', default=False)
@click.option('--image_o', type=click.INT, default=240)
@click.option('--alpha', type=click.FLOAT, default=1.0)
@click.option('--beta', type=click.FLOAT, default=0.15)
@click.option('--image-shape',
              nargs=2,
              type=click.Tuple([int, int]),
              default=(1000, 1000))
def main(image_folder, output_folder, eosin_hematoxylin, image_o, alpha, beta,
         image_shape):
    image_folder = Path(image_folder).resolve()
    output_folder = Path(output_folder).resolve()
    files = [f for f in image_folder.rglob("*.tif")]

    for img_path in tqdm(files):

        img = get_array_from_tiff(img_path, image_shape=image_shape)

        img_normalize, img_hematoxylin, img_eosin = normalizeStaining(
            img=img,
            Io=image_o,
            alpha=alpha,
            beta=beta,
        )
        Image.fromarray(img_normalize).save(
            str(output_folder / (img_path.stem + ".tiff")))

        if eosin_hematoxylin:
            Image.fromarray(img_hematoxylin).save(
                str(output_folder / (img.stem + "_hematoxylin.png")))
            Image.fromarray(img_eosin).save(
                str(output_folder / (img.stem + "_eosin.png")))


def get_array_from_tiff(img_path, image_shape=(1000, 1000)):
    im = Image.open(str(img_path))
    for page in ImageSequence.Iterator(im):
        image = np.array(page)
        if image.shape[:2] == image_shape:
            return image

    raise RuntimeError(f"No image in {img_path} have the shape {image_shape}.")


def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    #eigvecs *= -1

    #project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(
        Io,
        np.exp(
            np.expand_dims(-HERef[:, 0],
                           axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(
        Io,
        np.exp(
            np.expand_dims(-HERef[:, 1],
                           axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return Inorm, H, E


if __name__ == '__main__':
    main()