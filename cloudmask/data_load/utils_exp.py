import os
import numpy as np
from keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def load_test_data(path, file="test"):
    """
    Load the test data given a path of a given train/test split
    :param path:
    :return:
    """
    x_test, y_test = np.load(os.path.join(path, f"x_{file}.npy")), np.load(
        os.path.join(path, f"y_{file}.npy")
    )

    return x_test, y_test.astype(str)


def load_train_data(path):
    """
    Load the training and validation data
    :param path (str): validation fold where the training data is saved
    :return:
    """

    x_train, y_train = np.load(os.path.join(path, "x_train.npy")), np.load(
        os.path.join(path, "y_train.npy")
    )

    x_val, y_val = np.load(os.path.join(path, "x_val.npy")), np.load(
        os.path.join(path, "y_val.npy")
    )

    return x_train, y_train.astype(str), x_val, y_val.astype(str)


def subset_data(x_train, y_train, groups=None, factor=5):
    """
    Subset the training data by a factor N (e.g. by 10, we will only keep 1/10 for the experiments)
    :param x_train:
    :param y_train:
    :param groups:
    :param factor:
    :return:
    """
    rdm_train = np.random.RandomState(0).choice(
        x_train.shape[0], x_train.shape[0] // factor
    )
    x_train, y_train = shuffle(
        x_train[
            rdm_train,
        ],
        y_train[
            rdm_train,
        ],
    )
    if groups is None:
        return x_train, y_train
    groups = groups[rdm_train]
    return x_train, y_train, groups


def get_training_ml_experiment(path_folder_test, n_folds=4):
    """
    For common ml methods, leave_one_group_out method requires a np.array with the observation associated group
    Therefore, we associate each validation set to one group
    :param path_folder_test (str) : path where a given test set is saved
    :param n_folds (int) : Number of validation folds for a given training set
    :return: training data (x_train and y_train) and its associated group of validation
    """

    x_train = []
    y_train = []
    val_group = []

    for fold in range(n_folds):
        path_val_fold = os.path.join(path_folder_test, f"fold_{fold + 1}")

        x_val, y_val = np.load(os.path.join(path_val_fold, "x_val.npy")), np.load(
            os.path.join(path_val_fold, "y_val.npy")
        )

        x_train.append(x_val)
        y_train.append(y_val)
        val_group.append(
            np.array([[str(fold + 1)] * x_val.shape[0]]).reshape(x_val.shape[0], 1)
        )

    return (
        np.concatenate(x_train, axis=0),
        np.concatenate(y_train, axis=0),
        np.concatenate(val_group, axis=0),
    )


def get_dictionary_metrics(y_true, y_preds, group=""):
    """
    Save a dictionary with all the metrics to monitor the experiments
    :param y_true (np.array): labeled pixels
    :param y_preds (np.array): predicted pixels
    :param group : attribute add to the dictionary
    :return:
    """

    def _check_shape(target):
        if len(target.shape) == 1:  # check if we had a flatten vector
            target = target.reshape(target.shape[0], 1)
        if (
            target.shape[1] <= 1
        ):  # if we have original labels : must set from 0 to n_class-1 to get the 6 dummies
            target = to_categorical(target, num_classes=6)

        return target

    y_true = _check_shape(y_true)
    y_preds = _check_shape(y_preds)

    return {
        "Acc": [accuracy_score(y_true, y_preds)],
        "Recall": [
            recall_score(y_true, y_preds, average="weighted")
        ],  # weighted to take into account class imbalance
        "Precision": [precision_score(y_true, y_preds, average="weighted")],
        "F1": [f1_score(y_true, y_preds, average="weighted")],
        "group": [group],
    }


def _reformat_labels(y):
    y = y.astype(int) // 10 - 1
    y = y.astype(str)
    return y


def get_categorical_labels(y_train, y_val, y_test):
    """
    Get labels ids into one hot encoded.
    Since we have 6 classes with 10 base coded, we divide by 10 and retrieve 1 to get 6 columns (specific to this database)
    :param y_train:
    :param y_val:
    :param y_test:
    :return:
    """
    y = np.concatenate([y_train, y_val, y_test], axis=0)
    y = to_categorical(y.astype(int) // 10 - 1)

    y_train, y_val, y_test = (
        y[
            : y_train.shape[0],
        ],
        y[
            y_train.shape[0] : (y_train.shape[0] + y_val.shape[0]),
        ],
        y[(y_train.shape[0] + y_val.shape[0]) :],
    )
    return y_train, y_val, y_test
