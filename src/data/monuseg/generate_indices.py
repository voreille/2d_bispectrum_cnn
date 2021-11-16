import json

import click
import numpy as np
from numpy.lib.type_check import _imag_dispatcher

from src.data.monuseg.tf_data import get_split


@click.command()
@click.option(
    "--filename",
    type=click.Path(),
    default=
    "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/indices/monuseg.json")
@click.option("--n-rep", type=click.INT, default=10)
def main(filename, n_rep):
    indices_list = list()

    for _ in range(n_rep):
        image_ids_train, image_ids_val, image_ids_test = get_split()
        indices_list.append({
            "train": image_ids_train,
            "val": image_ids_val,
            "test": image_ids_test,
        })

    with open(filename, "w") as f:
        json.dump(indices_list, f)


if __name__ == '__main__':
    main()
