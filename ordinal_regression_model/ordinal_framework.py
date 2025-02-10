from typing import List, Dict, Tuple

import numpy as np

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


def compute_joint_probability(proba_binary: np.ndarray) -> np.ndarray:
    """
    Compute joint probabilities from binary classifier outputs, including a null class.

    Given binary classifier outputs for the cumulative probabilities p(Y <= i)
    for each threshold (i.e. each column in proba_binary), this function computes
    the probability for class i (for i = 0,..., n_classes-1) as:

        P(Y = i) = (∏_{j=0}^{i-1} [1 - p(Y <= j)]) * (∏_{j=i}^{n_classes-1} p(Y <= j))

    An additional null class probability is computed as the remaining probability mass
    so that the sum over all classes (including the null class) is 1.

    Parameters
    ----------
    proba_binary : np.ndarray of shape (n_samples, n_classes - 1)
        The i-th column corresponds to the cumulative probability p(Y <= i).

    Returns
    -------
    proba : np.ndarray of shape (n_samples, n_classes + 1)
        The computed joint probabilities for each class (columns 0 to n_classes-1),
        with the last column (index n_classes) representing the null class probability.
    """
    n_samples, n_classes_minus_one = proba_binary.shape
    n_classes = n_classes_minus_one + 1

    # For each classifier, p(Y <= i)
    prob_bin_pos = proba_binary.copy()
    # For each classifier, 1 - p(Y <= i)
    prob_bin_neg = 1.0 - proba_binary

    # Allocate probability array with an extra column for the null class.
    proba = np.zeros((n_samples, n_classes + 1), dtype=float)

    # Class 0: no negative product term (empty product is 1).
    # P(Y = 0) = ∏_{j=0}^{n-1} p(Y <= j)
    proba[:, 0] = np.prod(prob_bin_pos, axis=1)

    # Classes 1 to n_classes-2 (i.e. for i = 1 to n-1, except the last regular class)
    for i in range(1, n_classes_minus_one):
        # P(Y = i) = [∏_{j=0}^{i-1} (1 - p(Y <= j))] * [∏_{j=i}^{n-1} p(Y <= j)]
        proba[:, i] = np.prod(prob_bin_neg[:, :i], axis=1) * np.prod(prob_bin_pos[:, i:], axis=1)

    # Last regular class, i = n_classes-1:
    # P(Y = n_classes-1) = ∏_{j=0}^{n-1} (1 - p(Y <= j))
    proba[:, n_classes_minus_one] = np.prod(prob_bin_neg, axis=1)

    # Compute the null class probability as the remaining probability mass.
    proba[:, n_classes] = 1.0 - np.sum(proba[:, :n_classes], axis=1)

    return proba
# =======================================================================================
def compute_joint_classification(
    proba_binary: np.ndarray,
    threshold: Tuple[float, ...]
) -> np.ndarray:
    """
    Compute the joint classification decision based on binary classifier probabilities and thresholds.

    Each column in `proba_binary` corresponds to the cumulative probability that Y <= i for the i-th classifier.
    For each binary classifier, a threshold is applied:
      - A positive decision is made if p(Y <= i) >= threshold[i].
      - A negative decision is made if p(Y <= i) < threshold[i].

    The final predicted class is determined by checking if the binary decisions follow one of these patterns:
      - Class 0 is predicted if all classifiers return a positive decision.
      - For an intermediate class i (1 <= i <= n_classifiers - 1), class i is predicted if:
            * The first i classifiers return a negative decision, and
            * The remaining classifiers return a positive decision.
      - The last regular class (index n_classifiers) is predicted if all classifiers return a negative decision.
      - If none of these patterns is satisfied, a null class is assigned (label -1).

    Parameters
    ----------
    proba_binary : np.ndarray of shape (n_samples, n_classes - 1)
        The cumulative probabilities for each binary classifier.
    threshold : Tuple[float, ...]
        A tuple of thresholds for each binary classifier; each threshold must lie in [0, 1].

    Returns
    -------
    pred_label : np.ndarray of shape (n_samples,)
        The final predicted class labels (with a label of -1 for the null class).
    """
    n_samples, n_class_minus_one = proba_binary.shape
    assert n_class_minus_one == len(threshold), "The number of thresholds should equal the number of binary classifiers."
    assert np.all(np.array(threshold) >= 0) and np.all(np.array(threshold) <= 1), "Each threshold must be in the range [0, 1]."

    # Total number of regular classes is one more than the number of binary classifiers.
    n_classes = n_class_minus_one + 1

    # Copy cumulative probabilities.
    proba_pos = proba_binary.copy()

    # Allocate boolean arrays for binary decisions.
    label_pos = np.zeros_like(proba_binary, dtype=bool)  # True if p(Y <= i) >= threshold[i]
    label_neg = np.zeros_like(proba_binary, dtype=bool)  # True if p(Y <= i) < threshold[i]

    # Apply thresholds for each classifier.
    for i in range(n_class_minus_one):
        label_pos[:, i] = proba_pos[:, i] >= threshold[i]
        label_neg[:, i] = proba_pos[:, i] <  threshold[i]

    # Create an array to hold the decision for each possible class.
    # We include an extra column for the null class.
    pred_label_bin = np.zeros((n_samples, n_classes + 1), dtype=bool)

    # Class 0: predicted if all classifiers yield a positive decision.
    pred_label_bin[:, 0] = np.all(label_pos, axis=1)

    # For intermediate classes 1 to n_class_minus_one - 1:
    # Predicted if the first i classifiers are negative and the remaining classifiers are positive.
    for i in range(1, n_class_minus_one):
        pred_label_bin[:, i] = np.all(label_neg[:, :i], axis=1) & np.all(label_pos[:, i:], axis=1)

    # Last regular class: predicted if all classifiers yield a negative decision.
    pred_label_bin[:, n_class_minus_one] = np.all(label_neg, axis=1)

    # Null class: predicted if none of the above conditions is met.
    pred_label_bin[:, n_classes] = ~np.any(pred_label_bin[:, :n_classes], axis=1)

    #
    assert np.all(np.sum(pred_label_bin, axis=1) == 1), "Each sample should be assigned to exactly one class."

    # Final predicted label is taken as the index of the first True value.
    # Convert the null class (index n_classes) to -1.
    pred_label = np.argmax(pred_label_bin.astype(int), axis=1)
    pred_label[pred_label == n_classes] = -1

    return pred_label
# ==========================================================================================================
class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self,
        estimator,
        estimator_params : List[Dict] = None,
        n_classes : int = 3,
        threshold : Tuple[float, ...] = (0.5, 0.5),
    ):
        """
        Initialize the OrdinalClassifier.

        Parameters
        ----------
        estimator : estimator object
            The base estimator used for each binary classification task.
        estimator_params : List[Dict]
            A list of parameter dictionaries for each binary estimator. Its length must equal n_classes - 1.
        n_classes : int, default=3
            The total number of ordinal classes.
        threshold : Tuple[float, ...], default=(0.5, 0.5)
            A tuple of thresholds for each binary classifier; each value must be in [0, 1].
            Its length must be equal to n_classes - 1.
        """
        assert (n_classes - 1) == len(threshold) == len(estimator_params), \
            'Class number "n_classes" should be equal to "threshold" number + 1.'
        assert all([0. <= t <= 1. for t in threshold]), '"threshold" values should be between 0 and 1.'

        self.n_classes = n_classes
        self.threshold = threshold

        self.estimator = estimator
        self.estimator_params = estimator_params

    # ---------------------------------------------------------------------
    def fit(self, X, y):
        """
        Fit the ordinal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,)
            Target labels (ordinal values).

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X, y = check_X_y(X, y)

        # Get unique classes
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        assert self.n_classes == self.n_classes_, (
            'The number of unique classes in y should equal n_classes.'
        )

        # Generate binary labels: for each i, y_binary[i] is 1 if y <= i and 0 otherwise.
        y_binary = [(y <= i).astype(int) for i in range(self.n_classes - 1)]

        # Fit one binary estimator per threshold.
        self.binary_estimators_ = []
        for ix, _y in enumerate(y_binary):
            _est = clone(self.estimator)
            _est.set_params(**self.estimator_params[ix])
            _est.fit(X, _y)
            self.binary_estimators_.append(_est)

        return self
    # -------------------------------------------------------------------------------
    def _predict_proba_binary(self, X):
        """
        Predict the probabilities for the positive class from each binary classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba_binary : np.ndarray of shape (n_samples, n_classes - 1)
            The predicted probabilities that Y <= i for each classifier.
        """
        return np.array([est.predict_proba(X)[:, 1] for est in self.binary_estimators_]).T
    # -------------------------------------------------------------------------------
    def predict(self, X):
        """
        Predict ordinal class labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        pred_label : np.ndarray of shape (n_samples,)
            Predicted class labels. A label of -1 indicates that none of the prescribed conditions were met.
        """
        # Validate input
        check_is_fitted(self)
        X = check_array(X)

        pred_proba_bin = self._predict_proba_binary(X)
        pred = compute_joint_classification(pred_proba_bin, self.threshold)
        return pred
    # -------------------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        pred_proba : np.ndarray of shape (n_samples, n_classes + 1)
            Predicted probabilities for each class (including the null class).
        """
        # Validate input
        check_is_fitted(self)
        X = check_array(X)

        # probability of N-1 binary classifiers
        pred_proba_bin = self._predict_proba_binary(X)
        pred_proba = compute_joint_probability(pred_proba_bin)
        return pred_proba
# ==========================================================================================================
