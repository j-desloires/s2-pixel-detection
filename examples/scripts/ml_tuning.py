import os
import sys

root_path = "/home/johann/git-repo/cloud-mask"
sys.path.insert(0, root_path)
sys.path.append("../../cloudmask/")  #

from sklearn.ensemble import RandomForestClassifier
from cloudmask.data_load.utils_exp import (
    load_test_data,
    subset_data,
    get_training_ml_experiment,
    _reformat_labels,
)
from cloudmask.base_models.base_ml_tuning import get_predictions
import lightgbm as lgb

##########################################################################################
###############
# RF
rf = RandomForestClassifier

max_depth = [int(x) for x in range(2, 12, 2)]
max_depth.append(None)

# Create the random grid
grid_rf = dict(
    n_estimators=[500],  # [int(x) for x in range(100, 1100, 200)],
    max_features=["auto", "sqrt"],
    max_depth=max_depth,
    n_jobs=[3],
)

###############
# LightGBM standard version
lgb_cl = lgb.LGBMClassifier
# Create the random grid

grid_lgbm = {
    "learning_rate": [0.001, 0.01, 0.1],
    "n_estimators": [500],
    "num_leaves": [8, 16, 32, 64],
    "max_depth": [5, 7, 9],
    "boosting_type": ["gbdt", "dart"],
    "objective": ["multiclass"],
    "metric": ["multi_error"],
    "random_state": [500],
    "colsample_bytree": [0.8],
    "subsample": [0.5],
    "reg_alpha": [0.15],
    "reg_lambda": [0.15],
}

############################################################################################
"""
suffix = "lgbm_model"
grid_params = grid_lgbm
ml_model = lgb_cl
"""

suffix = "rf_model"
grid_params = grid_rf
ml_model = rf

ls_experiments = [
    # "experiments_continents",
    # "experiments_tiles",
    "experiments_countries",
]

# Take only 1/10 of the training observations (lack of computational power ..)
factor_subset = 10

for ls_experiment in ls_experiments:
    path_experiments = os.path.join(root_path, ls_experiment)

    test_folds = os.listdir(path_experiments)

    for test_folder in test_folds:

        folder_test = os.path.join(path_experiments, test_folder)
        x_test, y_test = load_test_data(folder_test)
        y_test = _reformat_labels(y_test)

        # Load training data for a given train/test split and the associated group for leave-one-group-out cross validation.
        x_train, y_train, val_groups = get_training_ml_experiment(
            folder_test, n_folds=4
        )
        # Subset the training data (lack of computational power ..)
        X, y, groups = subset_data(x_train, y_train, val_groups, factor=factor_subset)
        y = _reformat_labels(y)

        # Define the parameters regarding the training data ,(train/val/test, path, groups)
        params_prediction = dict(
            x_train=X,
            y_train=y.flatten(),
            x_test=x_test,
            y_test=y_test.flatten(),
            groups=groups.flatten(),
            path_save=folder_test,
        )

        get_predictions(
            model=ml_model,
            grid=grid_params,
            n_iter_search=50,  # search randomly 50 combination of hyperparameters
            suffix=suffix,
            **params_prediction
        )
