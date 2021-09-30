from random import shuffle
import json

import click
import numpy as np
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.model_selection import RepeatedKFold


@click.command()
@click.option(
    "--filename",
    type=click.Path(),
    default=
    "/home/valentin/python_wkspce/2d_bispectrum_cnn/data/indices/drive.json")
@click.option("--n-rep", type=click.INT, default=3)
@click.option("--n-splits", type=click.INT, default=4)
@click.option("--n-val", type=click.INT, default=5)
def main(filename, n_splits, n_rep, n_val):
    indices_list = list()

    indices = np.array([k for k in range(21, 41)])
    rkf = RepeatedKFold(n_splits=n_splits,
                        n_repeats=n_rep,
                        random_state=2652124)
    for idx_train, idx_test in rkf.split(indices):

        indices_list.append({
            "train": [int(indices[i]) for i in idx_train[n_val:]],
            "val": [int(indices[i]) for i in idx_train[:n_val]],
            "test": [int(indices[i]) for i in idx_test],
        })

    with open(filename, "w") as f:
        json.dump(indices_list, f)


if __name__ == '__main__':
    main()
