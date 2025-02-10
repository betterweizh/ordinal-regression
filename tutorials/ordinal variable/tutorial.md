Ordinal Variable

# 1. Define

- Ordinal variable is a type of categorical variable that has a clear order or ranking.

  - i.e., unlike < neutral < somewhat like < very like

# 2. Processing oridnal variable by using `pandas` package

## 2.1 `CategoricalDtype` : categorical data type


- create a categorical data type

```python
import pandas as pd
from pandas.api.types import CategoricalDtype
# define a categorical data type
# Note: this is a instance of data type (like 'float', 'int'), not a series
cat_type = CategoricalDtype(categories=['a', 'b', 'c'], ordered=True)
# or 
cat_type = pd.CategoricalDtype(categories=['a', 'b', 'c'], ordered=True)

print(type(cat_type))
# <class 'pandas.core.dtypes.dtypes.CategoricalDtype'>
```

- create a `Series` instance with the categorical data type

```python
# create a series with the categorical data type
s = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'], dtype=cat_type)
# or
s = pd.Series(['a', 'b', 'c', 'a', 'b', 'c']).astype(cat_type)
# or
s = pd.Series(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], categories=['a', 'b', 'c'], ordered=True))

print(type(s))
# <class 'pandas.core.series.Series'>
```

- create a categorical array by using codes

    - the *codes* indicate the index of the category in the `categories` list

```python
arr = pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=pd.CategoricalDtype(['a', 'b'], ordered=True))

print(type(arr))
# <class 'pandas.core.arrays.categorical.Categorical'>
```

## 2.2 Properties 

- properties of a `Series` with the categorical data type

```python
s.dtype
# Out: 'category'
s.values
# Reture a instance of 'pandas.core.arrays.categorical.Categorical'
s.cat
# Return a instance of 'pandas.core.arrays.categorical.CategoricalAccessor'
s.cat.categories
s.cat.ordered
s.cat.codes
```

- properties of a `Categorical` instance

```python
arr = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], categories=['a', 'b', 'c'], ordered=True)
# an instance of 'pandas.core.arrays.categorical.Categorical'
arr.categories
# Return an Index instance indicating all categories
arr.ordered
# Return a bool (True or False). Whether the categories have an order or not
arr.codes
# Return a Series instance of integers that indicate the index of the category in the `categories` list
```
