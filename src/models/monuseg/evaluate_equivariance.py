from pathlib import Path
import os

from tqdm import tqdm
import pandas as pd

from src.models.monuseg.evaluation import evaluate_equivariance
# from src.models.monuseg.models_old import get_model
from src.models.monuseg.models import get_model
from src.models.monuseg.train_model import config_gpu
from src.data.monuseg.utils import get_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config_gpu(20000)

project_dir = Path(__file__).resolve().parents[3]
# model_name = "BispectUnet__rotation_True__nh_8__n_train_-1__psize_60x60__20211202-173734"
model_name = "MaskedUnet__rotation_True__nh_0__n_train_-1__psize_60x60__20211212-222429"
model_path = project_dir / f"models/MoNuSeg/{model_name}"


def main():
    model = get_model(
        model_name="MaskedUnet",
        output_channels=3,
        n_harmonics=8,
        n_feature_maps=[8, 16, 32],
        radial_profile_type="disks",
    )
    results_df = pd.DataFrame()
    for split in tqdm(range(10)):
        model.load_weights(model_path / f"weights/split_{split}/final")
        ids_train, ids_val, ids_test = get_split(split)
        results_train = evaluate_equivariance(model, ids_train)
        results_val = evaluate_equivariance(model, ids_val)
        results_test = evaluate_equivariance(model, ids_test)

        results_train["set"] = "train"
        results_test["set"] = "test"
        results_val["set"] = "val"
        df = pd.concat([results_train, results_test, results_val],
                       ignore_index=True)
        df["split"] = split
        results_df = results_df.append(df, ignore_index=True)

    results_df.to_csv(project_dir / f"results/equivariance/{model_name}.csv")


if __name__ == '__main__':
    main()