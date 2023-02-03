import os
import sys

# TODO : Add python command lines in readme with path and model specifications
root_path = "/home/johann/git-repo/cloud-mask"

sys.path.insert(0, root_path)
sys.path.append("../../cloudmask/")  #

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from cloudmask.ml_task import convnet_models
from cloudmask.data_load.utils_exp import (
    load_test_data,
    load_train_data,
    subset_data,
    get_categorical_labels,
)

#####################################################################################################

ls_experiments = [
    "experiments_tiles",
    "experiments_countries",
    "experiments_continents",
]

path_experiments = os.path.join(root_path, ls_experiments[0])

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
    loss="cross_entropy",  # "focal_loss", "cross_entropy"
    metrics="precision",
)

model_compiled = convnet_models.CNN1D(model_cfg_cnn)
model_compiled.prepare()

###########################################################################################
###########################################################################################
# Train the model
test_folds = os.listdir(path_experiments)

n_val_folds = 4
factor_subset = 10


for test_folder in test_folds:
    folder_test = os.path.join(path_experiments, test_folder)
    x_test, y_test_ = load_test_data(folder_test)
    for fold in range(n_val_folds):

        path_val_fold = os.path.join(folder_test, f"fold_{str(fold+1)}")
        x_train, y_train, x_val, y_val = load_train_data(path_val_fold)

        x_train, y_train = subset_data(x_train, y_train, factor=factor_subset)
        x_val, y_val = subset_data(x_val, y_val, factor=factor_subset)
        # One-hot labels for deep learning methods
        y_train, y_val, y_test = get_categorical_labels(y_train, y_val, y_test_)

        model_compiled.fit(
            train_dataset=(x_train.reshape(x_train.shape[0], 13, 1), y_train),
            val_dataset=(x_val.reshape(x_val.shape[0], 13, 1), y_val),
            batch_size=32,
            num_epochs=50,
            model_directory=os.path.join(
                path_val_fold, f"cnn_1D_enc_task_{model_compiled.config['loss']}"
            ),
            save_steps=2,  # check if best model each 2 epoch
        )

###########################################################################################
###########################################################################################
