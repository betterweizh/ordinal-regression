import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.miscmodels.ordinal_model import OrderedModel

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
    the MNLogit model (multinomial logistic regression).

    Parameters
    ----------
    ordinal_target : bool, default False
        If True, the target variable is treated as ordinal and converted to an
        ordered categorical variable using `ordinal_labels`.
    ordinal_labels : list-like, optional
        The ordered labels to use when converting the target variable. Must be provided
        if `ordinal_target` is True. The length must be at least 3.
    distribution : str, default 'logit'
        The distribution used in the ordinal model. Typically 'logit' or 'probit'.
    **fit_params : dict
        Additional keyword arguments passed to the model's fit method.
    """

    def __init__(self,
                 ordinal_target=False,
                 ordinal_labels=None,
                 distribution='logit',
                 **fit_params):
        # If target is ordinal, ensure that ordinal_labels are provided and valid.
        if ordinal_target:
            assert ordinal_labels is not None, (
                'Ordered labels must be provided to create an ordinal target when "ordinal_target" is True'
            )
            assert len(ordinal_labels) >= 3, "The number of ordered labels must be at least 3."

        self.ordinal_target = ordinal_target
        self.ordinal_labels = ordinal_labels
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
            pd.Categorical(y, categories=self.ordinal_labels, ordered=True),
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
        # Ensure input types: X must be a DataFrame and y must be a Series.
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."
        assert isinstance(y, pd.Series), "y must be a pandas Series."

        # Preserve original indices and feature names.
        self._sample_index = X.index
        self._feature_names = X.columns.to_list()

        # Validate X and y using scikit-learn utilities.
        X, y = check_X_y(X, y, y_numeric=True)
        # Reconstruct DataFrame and Series with the original indices and column names.
        X = pd.DataFrame(X, index=self._sample_index, columns=self._feature_names)
        y = pd.Series(y, index=self._sample_index)

        # Determine unique class labels and number of classes.
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Ordinal logistic regression requires at least 3 classes.
        assert self.n_classes_ >= 3, "The number of classes must be at least 3 for ordinal logistic regression."

        # If ordinal_target is True, convert y to an ordered categorical variable.
        if self.ordinal_target:
            # Ensure that the provided ordered labels match the unique classes.
            assert len(self.classes_) == len(self.ordinal_labels), (
                "The number of classes must match the number of ordered labels."
            )
            y_ord = self._convert_target(y)
            # Initialize the OrderedModel with the ordered categorical target.
            self.model_ = OrderedModel(endog=y_ord, exog=X, distr=self.distribution)
        else:
            # Otherwise, use MNLogit for multinomial logistic regression.
            self.model_ = MNLogit(endog=y, exog=X)

        # Fit the model using any additional parameters provided.
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
        # Ensure the model has been fitted.
        check_is_fitted(self)
        # Predict probabilities using the fitted model and return as a numpy array.
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
        # Ensure the model has been fitted.
        check_is_fitted(self)
        # Obtain predicted probabilities.
        proba = self.predict_proba(X)
        # Return the class with the highest predicted probability.
        return np.argmax(proba, axis=1)
