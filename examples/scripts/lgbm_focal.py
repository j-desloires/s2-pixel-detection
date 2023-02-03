import os
import sys
import pickle
import itertools

root_path = "/home/johann/git-repo/cloud-mask"

sys.path.insert(0, root_path)
sys.path.append("../../cloudmask/")  #

import numpy as np

from cloudmask.data_load.utils_exp import (
    load_test_data,
    subset_data,
    get_training_ml_experiment,
    _reformat_labels,
)
from cloudmask.base_models.base_ml_tuning import get_dictionary_metrics
from cloudmask.ml_task.ml_models import OneVsRestLightGBMWithCustomizedLoss, FocalLoss

############################################################################################
# Launch experiments for lgbm with focal loss

suffix = "custom_lgbm_model"

ls_experiments = [
    # "experiments_tiles",
    # "experiments_continents",
    "experiments_countries",
]

#  Try to tune focal loss (using validation sets)

grid = {"alpha": [0.25, 0.5, 0.75, 0.1], "gamma": [2, 0, 1]}
factor = 10

for alpha, gamma in itertools.product(*grid.values()):

    loss = FocalLoss(alpha=alpha, gamma=gamma)
    model = OneVsRestLightGBMWithCustomizedLoss(loss=loss)

    for ls_experiment in ls_experiments:
        path_experiments = os.path.join(root_path, ls_experiment)

        test_folds = os.listdir(path_experiments)

        for test_folder in test_folds:

            folder_test = os.path.join(path_experiments, test_folder)

            x_test, y_test = load_test_data(folder_test)
            y_test = _reformat_labels(y_test)

            x_train, y_train, val_groups = get_training_ml_experiment(
                folder_test, n_folds=4
            )
            # Subset the training data (lack of computational power ..)
            X, y, groups = subset_data(
                x_train, y_train, groups=val_groups, factor=factor
            )
            y = _reformat_labels(y)

            # Tune with validation set

            for subgroup in np.unique(groups):
                idx_val = np.where(groups == subgroup)[0]
                X_train, y_train = (
                    X[
                        ~idx_val,
                    ],
                    y[~idx_val],
                )
                X_val, y_val = (
                    X[
                        idx_val,
                    ],
                    y[idx_val],
                )

                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)

                dict_metrics = get_dictionary_metrics(
                    y_val,
                    y_val_pred,
                    group={"alpha": alpha, "gamma": gamma, "fold": subgroup},
                )

                with open(
                    os.path.join(
                        folder_test,
                        f"dict_val_metrics_{suffix}_{subgroup}_{alpha}_{gamma}.pickle",
                    ),
                    "wb",
                ) as d:
                    pickle.dump(dict_metrics, d, protocol=pickle.HIGHEST_PROTOCOL)

            # Fit overall training data and evaluate on test
            model.fit(X, y)
            y_test_pred = model.predict(x_test)

            dict_metrics = get_dictionary_metrics(
                y_test,
                y_test_pred,
                group={"alpha": alpha, "gamma": gamma},
            )
            dict_metrics["y_preds"] = y_test_pred

            with open(
                os.path.join(
                    folder_test, f"dict_test_metrics_{suffix}_{alpha}_{gamma}.pickle"
                ),
                "wb",
            ) as d:
                pickle.dump(dict_metrics, d, protocol=pickle.HIGHEST_PROTOCOL)
