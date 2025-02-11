import copy
from typing import List, Dict, Tuple

import tqdm
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import (
    precision_score, recall_score,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss)
from sklearn.utils.class_weight import compute_sample_weight

# ------------------------------------------------------------------
#     Ranked Probability Score (RPS)
# ------------------------------------------------------------------
def ranked_probability_score(y_true, y_prob, sample_weight=None):
    """
    Compute the Ranked Probability Score (RPS) for probabilistic forecasts,
    with an option to incorporate sample weights.

    The RPS is defined as the weighted average (or simple average, if no weights are provided)
    of the sum over classes of the squared differences between the cumulative predicted
    probability distribution and the cumulative observed distribution. Lower scores indicate
    better forecast performance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels. These should be consecutive integers (e.g., 0, 1, 2, ..., K-1).
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probability distribution over classes for each sample.
        Each row is expected to represent a valid probability distribution (summing to 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If provided, the RPS is computed as:
            RPS = sum_i (sample_weight[i] * (sum_j (cumulative_error[i, j])))/sum(sample_weight)
        Otherwise, a simple average over samples is computed.

    Returns
    -------
    rps : float
        The Ranked Probability Score.

    Raises
    ------
    AssertionError
        If y_true does not contain consecutive class labels starting at 0 or if the number
        of unique classes in y_true does not match the number of columns in y_prob.
    """
    # Identify unique classes and verify they are consecutive starting from 0.
    classes = np.unique(y_true)
    assert len(classes) == (max(classes) + 1), "y_true must contain consecutive class labels starting from 0."
    assert len(classes) == y_prob.shape[1], (
        "The number of unique classes in y_true must equal the number of columns in y_prob."
    )

    n_samples = len(y_true)
    n_classes = len(classes)

    # Convert y_true to one-hot encoding.
    y_true_onehot = np.zeros((n_samples, n_classes), dtype=float)
    y_true_onehot[np.arange(n_samples), y_true] = 1.0

    # Compute the cumulative true distribution.
    # For example, for a true label 1 in a 3-class problem, the cumulative vector is [0, 1, 1].
    y_true_cum = np.cumsum(y_true_onehot, axis=1)

    # Compute the cumulative predicted probabilities.
    y_prob_cum = np.cumsum(y_prob, axis=1)

    # Compute the squared errors between the cumulative true and predicted distributions.
    # This yields an (n_samples, n_classes) error matrix.
    errors = (y_true_cum - y_prob_cum) ** 2

    if sample_weight is not None:
        # Ensure sample_weight is a numpy array.
        sample_weight = np.asarray(sample_weight)
        # Compute the weighted error per sample, then sum over samples.
        total_error = np.sum(np.sum(errors, axis=1) * sample_weight)
        rps = total_error / np.sum(sample_weight)
    else:
        # Compute the unweighted average error per sample (summing errors across classes).
        rps = np.sum(errors) / n_samples

    return rps
# ========================================================================================================
def classification_metric(y_true, y_pred, y_pred_prob, sample_weight='balanced'):
    """"
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    y_pred_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    sample_weight : str or array-like of shape (n_samples,), default='balanced'
        Sample weights. If 'balanced', class weights are computed using the balanced strategy.

    Returns
    -------
    metrics : dict
        A dictionary containing the computed metrics.
    """
    if isinstance(sample_weight, str):
        if sample_weight == 'balanced':
            sample_weight = compute_sample_weight('balanced', y_true)
    elif isinstance(sample_weight, np.ndarray):
        assert len(sample_weight) == len(y_true)
    else:
        raise ValueError('Invalid sample_weight value. Must be "balanced" or an array-like object.')

    metrics = {
        'precision'                 : precision_score(y_true, y_pred, average='macro'),
        'recall'                    : recall_score(y_true, y_pred, average='macro'),
        # For 'roc_auc' Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes
        'roc_auc'                   : roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr'),
        # 'average_precision'         : average_precision_score(y_true, y_pred_prob, average='macro', multi_class='ovr'),
        'accuracy'                  : accuracy_score(y_true, y_pred),
        'balanced_accuracy'         : balanced_accuracy_score(y_true, y_pred),
        'f1_macro'                  : f1_score(y_true, y_pred, average='macro'),
        'cohen_kappa'               : cohen_kappa_score(y_true, y_pred),
        'log_loss'                  : log_loss(y_true, y_pred_prob, normalize=True),
        # 'balanced_log_loss'         : log_loss(y_true, y_pred_prob, normalize=True, sample_weight=sample_weight),
        'ranked_probability_score'  : ranked_probability_score(y_true, y_pred_prob),
        # 'balanced_ranked_probability_score' : ranked_probability_score(y_true, y_pred_prob, sample_weight=sample_weight),
    }
    return metrics
# ========================================================================================================


# ---------------------------------------------------------------------------
#   metrics with new classes in prediction (not appeared in y_true)
# ---------------------------------------------------------------------------
def filter_nan_value(y_true, y_pred, nan_value=-1):
    """
    Filter out samples where the predicted label equals the specified nan_value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels, where a value equal to nan_value indicates a missing prediction.
    nan_value : int or float, default=-1
        The value used to represent missing predictions.

    Returns
    -------
    y_true_filtered : array-like
        The true labels corresponding to valid predictions.
    y_pred_filtered : array-like
        The predicted labels corresponding to valid predictions.
    n_nan : int
        The number of samples with missing predictions.
    """
    mask = y_pred != nan_value
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    n_nan = np.sum(~mask)
    return y_true_filtered, y_pred_filtered, n_nan
# ----------------------------------------------------------------------------------------------------------------------
def valid_sample_proportion(y_true, y_pred, nan_value=-1):
    """
    Compute the proportion of samples that have valid (non-nan) predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    nan_value : int or float, default=-1
        The value representing missing predictions.

    Returns
    -------
    proportion : float
        The proportion of samples with valid predictions.
    """
    n_samples = len(y_true)
    n_nan = np.sum(y_pred == nan_value)
    return (n_samples - n_nan) / n_samples
# ----------------------------------------------------------------------------------------------------------------------
def balanced_accuracy_with_nan(y_true, y_pred, nan_value=-1):
    """
    Compute the balanced accuracy score, adjusted for samples with missing predictions.

    Missing predictions (indicated by nan_value) are removed before computing the balanced accuracy,
    and the resulting score is scaled by the proportion of valid samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    nan_value : int or float, default=-1
        The value representing missing predictions.

    Returns
    -------
    score : float
        The balanced accuracy score adjusted for missing predictions.
    """
    n_samples = len(y_true)
    y_true_valid, y_pred_valid, n_nan = filter_nan_value(y_true, y_pred, nan_value)
    score = balanced_accuracy_score(y_true_valid, y_pred_valid)
    valid_prop = (n_samples - n_nan) / n_samples
    return valid_prop * score
# ----------------------------------------------------------------------------------------------------------------------
def f1_with_nan(y_true, y_pred, nan_value=-1):
    """
    Compute the macro-averaged F1 score, adjusted for samples with missing predictions.

    Missing predictions (indicated by nan_value) are removed before computing the F1 score,
    and the resulting score is scaled by the proportion of valid samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    nan_value : int or float, default=-1
        The value representing missing predictions.

    Returns
    -------
    score : float
        The macro-averaged F1 score adjusted for missing predictions.
    """
    n_samples = len(y_true)
    y_true_valid, y_pred_valid, n_nan = filter_nan_value(y_true, y_pred, nan_value)
    score = f1_score(y_true_valid, y_pred_valid, average='macro')
    valid_prop = (n_samples - n_nan) / n_samples
    return valid_prop * score
# ----------------------------------------------------------------------------------------------------------------------
def _cls_metric(y_true, y_pred, y_pred_prob, sample_weight):
    metrics = {
        'precision'                 : precision_score(y_true, y_pred, average='macro'),
        'recall'                    : recall_score(y_true, y_pred, average='macro'),
        'accuracy'                  : accuracy_score(y_true, y_pred),
        'balanced_accuracy'         : balanced_accuracy_score(y_true, y_pred),
        'f1_macro'                  : f1_score(y_true, y_pred, average='macro'),
        'cohen_kappa'               : cohen_kappa_score(y_true, y_pred),
        'log_loss'                  : log_loss(y_true, y_pred_prob, normalize=True),
        # 'balanced_log_loss'         : log_loss(y_true, y_pred_prob, normalize=True, sample_weight=sample_weight),
        'ranked_probability_score'  : ranked_probability_score(y_true, y_pred_prob),
        # 'balanced_ranked_probability_score' : ranked_probability_score(y_true, y_pred_prob, sample_weight=sample_weight),
    }
    return metrics
# ========================================================================================================
def classification_metric_nan(y_true, y_pred, y_pred_prob, nan_value=-1):
    """
    Compute classification metrics while handling missing predictions.

    Missing predictions are indicated by y_pred values equal to nan_value. This function
    filters out such samples, computes classification metrics on the remaining (valid) samples,
    and then scales the resulting metrics by the proportion of valid samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels, where a value equal to nan_value indicates a missing prediction.
    y_pred_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities for each sample.
    nan_value : int or float, default=-1
        The value representing missing predictions.

    Returns
    -------
    metrics : dict
        A dictionary containing the computed metrics for the valid samples (e.g., accuracy,
        balanced accuracy, etc.), scaled by the valid sample proportion. Also includes the key
        'valid_sample_proportion'.
    """
    assert len(np.unique(y_true)) == y_pred_prob.shape[1], (
        "The number of unique classes in y_true must match the number of columns in y_pred_prob."
    )

    # Create a boolean mask for valid predictions (i.e., predictions that are not nan_value).
    mask = y_pred != nan_value

    # Compute the total number of samples and the number of missing predictions.
    n_samples = len(y_true)
    n_nan = np.sum(~mask)
    valid_prop = (n_samples - n_nan) / n_samples

    if valid_prop < 0.5:
        # If all predictions are missing, return a dictionary of NaN values.
        return {'valid_sample_proportion' : valid_prop}

    # Filter out samples with missing predictions using explicit 2D indexing for y_pred_prob.
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    y_pred_prob_valid = y_pred_prob[mask, :]

    # Compute balanced sample weights for the entire dataset and then filter them.
    sample_weight = compute_sample_weight('balanced', y_true)
    sample_weight_valid = sample_weight[mask]

    # Compute classification metrics for the valid samples.
    # (Assumes that the function `classification_metric` is defined elsewhere.)
    metrics = _cls_metric(
        y_true_valid, y_pred_valid, y_pred_prob_valid, sample_weight=sample_weight_valid
    )

    # Scale each metric by the proportion of valid samples.
    # (Note: log_loss is not scaled by the valid proportion.)
    for k, v in metrics.items():
        if k not in ['log_loss']:
            metrics[k] = v * valid_prop
    metrics['valid_sample_proportion'] = valid_prop

    return metrics
# ========================================================================================================

