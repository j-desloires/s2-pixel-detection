import os
import numpy as np
from cloudmask.data_load.utils_exp import (
    load_test_data,
)

from sklearn.metrics import f1_score, confusion_matrix
from cloudmask.ml_task import convnet_models
import seaborn as sns
import matplotlib.pyplot as plt

#####################################################################################################
target_names = ["Clear", "Water", "Snow", "Cirrus", "Cloud", "Shadow"]

# Model configuration CNN

model_cfg_cnn = dict(
    learning_rate=10e-4,
    keep_prob=0.5,
    keep_prob_conv=0.8,
    nb_conv_filters=32,
    nb_conv_stacks=3,
    nb_fc_neurons=32,
    nb_fc_stacks=1,
    fc_activation="relu",
    kernel_size=3,
    padding="SAME",
    emb_layer="GlobalAveragePooling1D",
    enumerate=True,
    str_inc=True,
    n_classes=6,
    kernel_regularizer=1e-8,
    loss="focal_loss",  # "focal_loss", "cross_entropy" => switch in model_config to load the weights
    metrics="precision",
)

model_compiled = convnet_models.CNN1D(model_cfg_cnn)
model_compiled.prepare()


##################################################################################
# Compute average and standard error from each test fold
root_path = "/home/johann/git-repo/cloud-mask"

ls_experiments = [
    "experiments_tiles",
    "experiments_countries",
    "experiments_continents",
]
exp = ls_experiments[2]
f_metrics = []

for test_id in range(1, 6):
    # Load a test data
    path_experiments = os.path.join(root_path, exp)

    folder_test = os.path.join(path_experiments, f"test_{test_id}")
    # Read the associated labels
    x_test, y_test = load_test_data(folder_test)
    x_test = x_test.reshape(x_test.shape[0], 13, 1)
    # Instantiate the cnn model with just one observation
    _ = model_compiled(x_test[:1,])

    ######################################################################
    # Search model which performs the best on a given validation fold
    metrics_fold = []

    for val_folds in range(1, 5):
        # Load weights from deep learning methods
        path_val_fold = os.path.join(folder_test, f"fold_{val_folds}")
        model_compiled.encoder.load_weights(
            os.path.join(
                path_val_fold,
                f"cnn_1D_enc_task_{model_compiled.config['loss']}/encoder_best_model",
            )
        )

        model_compiled.task.load_weights(
            os.path.join(
                path_val_fold,
                f"cnn_1D_enc_task_{model_compiled.config['loss']}/task_best_model",
            )
        )
        x_val, y_val = load_test_data(path_val_fold, file="val")
        enc_predicted = model_compiled.encoder.predict_on_batch(
            x_val.reshape(x_val.shape[0], 13, 1)
        )
        task_predicted = model_compiled.task.predict_on_batch(enc_predicted)
        task_predicted = (np.argmax(task_predicted, axis=1) + 1) * 10
        metric_val = f1_score(
            y_val.flatten(), task_predicted.astype(str), average="weighted"
        )
        metrics_fold.append(metric_val)

    max_val = np.argmax(metrics_fold) + 1
    path_val_fold = os.path.join(folder_test, f"fold_{max_val}")
    ######################################################################
    # Load weights from best model and predict on test set
    model_compiled.encoder.load_weights(
        os.path.join(
            path_val_fold,
            f"cnn_1D_enc_task_{model_compiled.config['loss']}/encoder_best_model",
        )
    )

    model_compiled.task.load_weights(
        os.path.join(
            path_val_fold,
            f"cnn_1D_enc_task_{model_compiled.config['loss']}/task_best_model",
        )
    )

    enc_predicted = model_compiled.encoder.predict_on_batch(x_test)
    task_predicted = model_compiled.task.predict_on_batch(enc_predicted)

    #############################################################
    labels_predict = np.argmax(task_predicted, axis=1)
    preds = (labels_predict + 1) * 10
    preds = preds.astype(str)

    metric_ = f1_score(y_test, preds, average="weighted")
    f_metrics.append(metric_)


np.mean(f_metrics)
np.std(f_metrics)


##########################################################################################################
# Confusion map for a given experiments

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
