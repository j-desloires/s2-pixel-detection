import os
import pickle

# =============================================================================
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
)

from sklearn.metrics import accuracy_score, make_scorer
from cloudmask.data_load.utils_exp import get_dictionary_metrics


def get_best_parameters_random_search(
    param_dist,
    x,
    y,
    model,
    groups,
    scoring=make_scorer(accuracy_score),
    n_iter_search=50,
):
    """

    Random hyperparameters search using sklearn LeaveOneGroupOut() function
    Input:
        param_space (dict) : dictionary of parameters distribution
        x (np.array) : Input dataset
        y (np.array) : target variable
        groups (np.array) : group name for each observation
        scoring = make_scorer(r2_score) : evaluation metric to minimize/maximize as make_scorer() object
        n_iter_search (int) : number of random search

    """

    if groups.shape[0] != y.shape[0]:
        raise ValueError(
            "Groups for LOGO validation must have same number of observations than the input data"
        )

    logo = LeaveOneGroupOut()

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring=scoring,
        cv=logo,
        n_jobs=-1,
    )

    random_search.fit(x, y, groups=groups)

    return random_search


def get_predictions(
    x_train,
    y_train,
    x_test,
    y_test,
    groups,
    model,
    grid,
    path_save,
    suffix,
    n_iter_search=50,
):

    model_to_tune = model()
    tuning = get_best_parameters_random_search(
        x=x_train,
        y=y_train,
        groups=groups,
        model=model_to_tune,
        param_dist=grid,
        n_iter_search=n_iter_search,
    )
    best_params = tuning.best_params_
    model_tuned = model(**best_params)
    model_tuned.fit(x_train, y_train)
    preds = model_tuned.predict(x_test)
    np.unique(y_test)
    #######
    # Save the metrics
    dict_metrics = get_dictionary_metrics(y_true=y_test, y_preds=preds, group="")
    dict_metrics["params"] = best_params
    # Save prediction to avoid to fit again the model
    dict_metrics["y_preds"] = preds
    print(dict_metrics)

    with open(os.path.join(path_save, f"dict_metrics_{suffix}.pickle"), "wb") as file:
        pickle.dump(dict_metrics, file)
