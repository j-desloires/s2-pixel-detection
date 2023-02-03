import os
import pickle
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from cloudmask.data_load.utils_exp import (
    load_test_data,
)


def _find_best_parameters_focal(folder_test, criterion="F1", grid=None):
    '''
    Find best combination of parameters for the focal loss given the experiments
    :param folder_test (str): path where the test folder is saved
    :param criterion (str) : criterion for the metric (F1, accuracy, precision or recall)
    :return (tuple) : best parameters given validation metrics
    '''

    if grid is None:
        grid = {"alpha": [0.1, 0.25, 0.5, 0.75], "gamma": [0, 1, 2]}

    results = {}

    for alpha, gamma in itertools.product(*grid.values()):
        for subgroup in range(1, 5):
            path_dict = os.path.join(
                folder_test,
                f"dict_val_metrics_{suffix}_{str(subgroup)}_{alpha}_{gamma}.pickle",
            )

            dict_val_metric = pickle.load(open(path_dict, "rb"))

            results[(alpha_, gamma_)] = dict_val_metric[criterion]

    results_df = pd.DataFrame(results).T.reset_index()
    id_best_val = np.argmax(results_df.iloc[:, -1].values)
    best_alpha, best_gamma = (
        results_df.iloc[id_best_val, 0],
        results_df.iloc[id_best_val, 1],
    )
    return best_alpha, best_gamma


##################################################################################
# Read results given a random test_fold and a kind of experiments

target_names = ["Clear", "Water", "Snow", "Cirrus", "Cloud", "Shadow"]

root_path = "/home/johann/git-repo/cloud-mask"
suffix = "custom_lgbm_model"

ls_experiments = [
    "experiments_tiles",
    "experiments_countries",
    "experiments_continents",
]

path_experiments = os.path.join(root_path, ls_experiments[2])

f_metrics = []


for test_id in range(1, 6):
    folder_test = os.path.join(path_experiments, f"test_{test_id}")
    # Read the associated labels
    x_test, y_test = load_test_data(folder_test)
    # Use standard values for focal loss. However, you can tune it using  _find_best_parameters_focal()
    best_alpha, best_gamma = 0.25, 2
    path_dict = os.path.join(
        folder_test,
        f"dict_test_metrics_{suffix}_{best_alpha}_{best_gamma}.pickle",
    )
    dict_test_metric = pickle.load(open(path_dict, "rb"))
    f_metrics.append(dict_test_metric["F1"])


np.mean(f_metrics)
np.std(f_metrics)


##############################################################################################
preds = (dict_test_metric["y_preds"] + 1) * 10
preds = preds.astype(str)

cm = confusion_matrix(y_test, preds, normalize="true")

ax = sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    xticklabels=target_names,
    yticklabels=target_names,
    cbar_kws={"label": "Accuracy %"},
)
ax.figure.axes[-1].yaxis.label.set_size(20)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show(block=False)
