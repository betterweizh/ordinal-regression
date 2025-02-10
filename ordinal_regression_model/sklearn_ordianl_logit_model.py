import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.miscmodels.ordinal_model import OrderedModel


class OrdinalLogitClassifier(ClassifierMixin, BaseEstimator):
    """
    A scikit-learn compatible classifier for ordinal logistic regression.

    This classifier wraps the OrderedModel from statsmodels to perform ordinal
    logistic regression. Alternatively, when `ordinal_target` is False, it uses
    the MNLogit model.

    Parameters
    ----------
    ordinal_target : bool, default False
        Whether the target variable is ordinal. If True, the target is converted to
        an ordered categorical variable using `ordinal_labels`.
    ordinal_labels : list-like, optional
        The ordered labels to use when converting the target variable. Must be provided
        if `ordinal_target` is True. The length must be at least 3.
    distribution : str, default 'logit'
        The distribution used in the ordinal model. Typically 'logit' or 'probit'.
    method : str, default 'bfgs'
        The optimization method used to fit the model.
    maxiter : int, default 1000
        Maximum number of iterations for the optimizer.
    disp : bool, default False
        If True, displays convergence messages during model fitting.
    **fit_params : dict
        Additional keyword arguments passed to the model's fit method.
    """
    def __init__(self,
                 ordinal_target=False,
                 ordinal_labels=None,
                 distribution='logit',
                 **fit_params):
        # When using ordinal target, ensure ordered labels are provided and there are at least 3 classes.
        if ordinal_target:
            assert ordinal_labels is not None, (
                'Ordered labels must be provided to create an ordinal target when "ordinal_target" is True'
            )
            assert len(ordinal_labels) >= 3, "The number of ordered labels must be at least 3."

        self.ordinal_target = ordinal_target
        self.ordered_labels = ordinal_labels
        self.distribution = distribution
        self.fit_params = fit_params

    def _convert_target(self, y):
        """
        Convert the target variable to an ordered categorical variable using the provided labels.

        Parameters
        ----------
        y : pd.Series
            The target variable.

        Returns
        -------
        y_ord : pd.Series
            An ordered categorical version of the target variable.
        """
        y_ord = pd.Series(
            pd.Categorical(y, categories=self.ordered_labels, ordered=True),
            index=y.index,
            name=y.name
        )
        return y_ord

    def fit(self, X, y):
        """
        Fit the ordinal logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The design matrix as a pandas DataFrame.
        y : array-like of shape (n_samples,)
            The target variable as a pandas Series. For ordinal regression, y should be encoded
            as integers or categories.

        Returns
        -------
        self : object
            The fitted classifier.
        """
        # Ensure X and y are provided as pandas DataFrame and Series, respectively.
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."
        assert isinstance(y, pd.Series), "y must be a pandas Series."

        # Preserve original indices and feature names.
        self._sample_index = X.index
        self._feature_names = X.columns.to_list()

        # Validate X and y.
        X, y = check_X_y(X, y, y_numeric=True)
        # Reconstruct DataFrame and Series with original indices and column names.
        X = pd.DataFrame(X, index=self._sample_index, columns=self._feature_names)
        y = pd.Series(y, index=self._sample_index)

        # Determine unique class labels and number of classes.
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Ensure there are at least 3 classes for ordinal logistic regression.
        assert self.n_classes_ >= 3, "The number of classes must be at least 3 for ordinal logistic regression."

        # If ordinal_target is True, convert y to an ordered categorical variable.
        if self.ordinal_target:
            # Ensure the provided ordered labels match the classes in y.
            assert len(self.classes_) == len(self.ordered_labels), (
                "The number of classes must match the number of ordered labels."
            )
            y_ord = self._convert_target(y)
            # Initialize the OrderedModel with ordered categorical target.
            self.model_ = OrderedModel(endog=y_ord, exog=X, distr=self.distribution)
        else:
            # Use the MNLogit model if not treating y as ordinal.
            self.model_ = MNLogit(endog=y, exog=X)

        # Fit the model using the specified optimization parameters.
        self.result_ = self.model_.fit(**self.fit_params)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The design matrix for which probabilities are predicted.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            The predicted probability distribution over ordinal classes.
        """
        # Ensure that the model has been fitted.
        check_is_fitted(self)
        # Use the fitted result to predict probabilities.
        return self.result_.predict(exog=X).values

    def predict(self, X):
        """
        Predict ordinal class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The design matrix for which class labels are predicted.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted ordinal class labels.
        """
        # Ensure that the model has been fitted.
        check_is_fitted(self)
        # Predict probabilities.
        proba = self.predict_proba(X)
        # Return the class with the highest predicted probability.
        return np.argmax(proba, axis=1)
# =========================================================================================================