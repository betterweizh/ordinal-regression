import copy
from typing import List, Dict, Tuple

import tqdm
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from .ordinal_framework import compute_joint_classification
from .metrics import classification_metric_nan


def grid_search_threshold(proba_bin, y_true, threshold_li, n_jobs=-1):
    """
    Perform a grid search over candidate threshold tuples to find the best threshold configuration.

    Each candidate threshold tuple is used to compute joint classification predictions via
    compute_joint_classification, and then evaluated against the true labels using a metric
    function (classification_metric_nan). The results (including the candidate thresholds and
    corresponding metrics) are collected into a DataFrame.

    Parameters
    ----------
    proba_bin : np.ndarray of shape (n_samples, n_classifiers)
        The predicted probabilities of positive for each binary classifier. Each value should be in [0, 1].
    y_true : array-like of shape (n_samples,)
        The true class labels.
    threshold_li : list of tuple of float
        A list where each element is a tuple of candidate thresholds for the binary classifiers.
        Each tuple must have length equal to the number of classifiers (i.e., proba_bin.shape[1])
        and each threshold must lie within the range [0, 1].
    n_jobs : int, default=-1
        The number of jobs to run in parallel (-1 means using all available processors).

    Returns
    -------
    res_all : pd.DataFrame
        A DataFrame containing the candidate threshold tuples and the corresponding classification
        metrics as computed by classification_metric_nan.
    """
    # Check that proba_bin and y_true have compatible shapes.
    assert proba_bin.shape[0] == len(y_true), "Number of samples in proba_bin and y_true must match."

    # number of classifiers
    n_classifiers = proba_bin.shape[1]

    # Ensure each candidate threshold tuple has the correct length.
    assert all(len(thres) == n_classifiers for thres in threshold_li), (
        'Each candidate threshold tuple must have length equal to the number of classifiers.'
    )
    # Ensure each threshold value is between 0 and 1.
    assert all(all(0.0 <= t <= 1.0 for t in thres) for thres in threshold_li), (
        '"threshold" values should be between 0 and 1.'
    )
    # Validate that all probabilities are in [0, 1].
    assert np.all((proba_bin >= 0.0) & (proba_bin <= 1.0)), 'The probabilities in proba_bin should be between 0 and 1.'

    # Define a helper function to evaluate one candidate threshold tuple.
    def evaluate_threshold(thres, proba_bin, y_true):
        res = {'threshold': thres}
        # compute_joint_classification is assumed to return a tuple:
        # (label_pos, label_neg, pred_label_bin, pred_label)
        y_pred = compute_joint_classification(proba_bin, threshold=thres)
        metric = classification_metric_nan(y_true, y_pred)
        res.update(metric)
        return copy.deepcopy(res)

    # Run evaluations in parallel with a progress bar.
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_threshold)(thres, proba_bin, y_true)
        for thres in tqdm.tqdm(threshold_li, desc="Grid search thresholds")
    )

    # Convert the list of result dictionaries into a DataFrame.
    results = pd.DataFrame(results)
    return results
# =====================================================================================
# def grid_search_cross_validation_threshold(model, X, y, param_grid, n_jobs=1, verbose=0, random_state=42):
#     '''
#     Example:
#     params_gird = {
#         'threshold' : [(np.around(i, 2), np.around(j, 2)) for i in np.arange(0, 1.05, 0.05) for j in np.arange(0, 1.05, 0.05)],}
#
#     cv_result = grid_search_cross_validation_threshold(
#         ordinal_model,
#         train_X, train_Y,
#         params_gird, n_jobs=31, verbose=1)
#     '''
#
#     grid_search = GridSearchCV(
#         model,
#         param_grid,
#         cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state),
#         scoring = {
#             'valid_sample_proportion'    : make_scorer(valid_sample_proportion, nan_value=-1),
#             'accuracy_with_nan'          : make_scorer(accuracy_with_nan, nan_value=-1),
#             'balanced_accuracy_with_nan' : make_scorer(balanced_accuracy_with_nan, nan_value=-1),
#             'f1_marco_with_nan'          : make_scorer(f1_with_nan, nan_value=-1),},
#         verbose = verbose,
#         n_jobs = n_jobs,
#         refit = False)
#
#     grid_search.fit(X, y)
#
#     return grid_search.cv_results_
# # =================================================================================================
