import pandas as pd
from pandas.api.types import is_integer_dtype, is_object_dtype, CategoricalDtype


def convert_label_to_ordinal(y, mapper, to_numeric=False):
    """
    Convert target labels to an ordered categorical variable using a mapper.

    Parameters
    ----------
    y : pd.Series
        The input target variable.
    mapper : dict
        A mapping from original label values to ordinal numeric values.
        For example: {'Low': 0, 'Medium': 1, 'High': 2}.
    to_numeric : bool, default False
        If True, map the target values using the mapper. Otherwise, use the original values.

    Returns
    -------
    pd.Series
        An ordered categorical Series with categories ordered according to the mapper.
    """
    # Ensure a mapper is provided.
    if mapper is None:
        raise ValueError("A mapper must be provided.")

    # Build the ordered list of categories by sorting the mapper items based on the ordinal values.
    ordered_labels = []
    ordered_values = []
    for _label, _num in sorted(mapper.items(), key=lambda x: x[1]):
        if _label not in ordered_labels:
            ordered_labels.append(_label)
            ordered_values.append(_num)


    # Optionally map the labels to their numeric ordinal values.
    if to_numeric:
        y_mapped = y.copy().map(mapper)
        ordered_categories = ordered_values
    else:
        y_mapped = y.copy()
        ordered_categories = ordered_labels

    # Convert the series to an ordered categorical type using the ordered categories.
    y_cat = pd.Series(
        pd.Categorical(y_mapped, categories=ordered_categories, ordered=True),
        index=y.index,
        name=y.name
    )
    return y_cat
# =====================================================================================
def convert_ordinal_to_label(y):
    """
    Convert an ordered categorical target variable to numerical codes.

    Parameters
    ----------
    y : pd.Series
        The target variable as an ordered categorical.

    Returns
    -------
    pd.Series
        A Series containing the numerical codes for the categorical values.
    """
    # Verify that the input is an ordered categorical variable.
    if (not isinstance(y.dtype, CategoricalDtype)) or (not y.cat.ordered):
        raise ValueError("Input y must be an ordered categorical variable.")

    # Return the categorical codes.
    y_num = pd.Series(y.cat.codes, index=y.index, name=y.name)
    return y_num
# =====================================================================================
def convert_target(y, object_type, mapper=None, **kwargs):
    """
    Convert the target variable between numerical and ordered categorical types.

    Parameters
    ----------
    y : pd.Series
        The target variable.
    object_type : str
        Desired output type:
          - 'ordinal' to convert numerical values to an ordered categorical variable.
          - 'int' to convert an ordered categorical variable to its numerical codes.
    mapper : dict, optional
        Mapping between original label values and ordinal numeric values.
        Required when converting numerical to ordered categorical.
    **kwargs :
        Additional keyword arguments to pass to convert_label_to_ordinal.

    Returns
    -------
    pd.Series
        The converted target variable.

    Raises
    ------
    ValueError
        If y is not of integer type or an ordered categorical, or if the conversion is unsupported.
    """
    # Ensure that y is either integer or an ordered categorical type.
    if not ( is_integer_dtype(y) or is_object_dtype(y) or isinstance(y.dtype, CategoricalDtype) ):
        raise ValueError("The target variable must be either of integer type or an ordered categorical.")

    # Convert numerical values to an ordered categorical variable.
    if ( is_integer_dtype(y) or is_object_dtype(y) ) and object_type == 'ordinal':
        if mapper is None:
            raise ValueError("A mapper must be provided when converting numerical to ordinal.")
        return convert_label_to_ordinal(y, mapper, **kwargs)

    # Convert an ordered categorical variable to its numerical codes.
    elif isinstance(y.dtype, CategoricalDtype) and object_type == 'label':
        return convert_ordinal_to_label(y)

    else:
        raise ValueError(
            "No conversion performed; the target variable is already of the desired type or the conversion is unsupported.")
# =====================================================================================
