import io
import zipfile
import pathlib
import pandas as pd

PROJECT_PATH = r'C:\Users\Wei Zhou\Documents\zhouwei file\Github-Project\ordinal-regression'
PROJECT_PATH = pathlib.Path(PROJECT_PATH)

def _load_wine_quality(preprocess=True, split_target=True):
    path = PROJECT_PATH / 'test_data/wine+quality.zip'

    with zipfile.ZipFile(path, 'r') as zf:
        wine_white = pd.read_csv(io.BytesIO(zf.read('winequality-white.csv')), header=0, sep=';', index_col=None) \
            .assign(color='white')
        wine_red = pd.read_csv(io.BytesIO(zf.read('winequality-red.csv')), header=0, sep=';', index_col=None) \
            .assign(color='red')

    data = pd.concat([wine_white, wine_red], axis=0, ignore_index=True)

    # preprocess the categorical variables
    # 'color' : white -> 0, red -> 1
    if preprocess:
        data['color'] = data['color'].map({'white': 0, 'red': 1})

    # split the target variable
    # target variable: 'quality'
    if split_target:
        target = data['quality']
        feature = data.drop(columns=['quality'])
        data = (feature, target)

    return data
# =============================================================================

def load_dataset(name, preprocess=True, split_target=True):
    if name == 'wine_quality':
        data = _load_wine_quality(preprocess, split_target)
    else:
        raise ValueError('Unknown dataset name: {}'.format(name))
    return data
# =============================================================================

