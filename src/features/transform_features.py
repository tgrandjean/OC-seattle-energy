"""transform features : transform the data for model.

This module contain functions needed to transform the data.
We want transform the data as following

Original data (Simplified for example)

+---------+-----------+-------------+-----------+---------+-----------+-------+
| Use 1   | Use 1 GFA | Use 2       | Use 2 GFA | Use 3   | Use 3 GFA | SiteE |
+---------+-----------+-------------+-----------+---------+-----------+-------+
| housing | 1234      | parking     | 567       | Nan     | Nan       | 4567  |
+---------+-----------+-------------+-----------+---------+-----------+-------+
| housing | 2345      | Nan         | Nan       | Nan     | Nan       | 5678  |
+---------+-----------+-------------+-----------+---------+-----------+-------+
| ...     | ...       | ...         | ...       | ...     | ...       | ...   |
+---------+-----------+-------------+-----------+---------+-----------+-------+
| office  | 3456      | data center | 2345      | parking | 1234      | 6789  |
+---------+-----------+-------------+-----------+---------+-----------+-------+

Data for model

+-------------+---------+--------+---------+--------------+
| data center | housing | office | parking | SiteEnegyUse |
+-------------+---------+--------+---------+--------------+
| 0           | 1234    | 0      | 567     | 4567         |
+-------------+---------+--------+---------+--------------+
| 0           | 2345    | 0      | 0       | 5678         |
+-------------+---------+--------+---------+--------------+
| ...         | ...     | ...    | ...     | ...          |
+-------------+---------+--------+---------+--------------+
| 2345        | 0       | 3456   | 1234    | 6789         |
+-------------+---------+--------+---------+--------------+

or

+-------------+---------+---------+---------+-------+--------------+
| data center | housing | office  | parking | Total | SiteEnegyUse |
+-------------+---------+---------+---------+-------+--------------+
| 0           | 0.68517 | 0       | 0.31482 | 1801  | 4567         |
+-------------+---------+---------+---------+-------+--------------+
| 0           | 1.00    | 0       | 0       | 2345  | 5678         |
+-------------+---------+---------+---------+-------+--------------+
| ...         | ...     | ...     | ...     |       | ...          |
+-------------+---------+---------+---------+-------+--------------+
| 0.33333     | 0       | 0.49125 | 0.17541 | 7035  | 6789         |
+-------------+---------+---------+---------+-------+--------------+

:note:
    Before running this script be sure to have the `full_data.pickle` file
    (see `notebooks/1.0-tg-initial-data-exploration.ipynb`) in the
    `data/interim` directory.

:usage:

.. code-block:: console

    (env)$python ./src/features/transform_features.py
"""
from pathlib import Path
import pandas as pd

CATEGORICAL_FEATURES = ('LargestPropertyUseType',
                        'SecondLargestPropertyUseType',
                        'ThirdLargestPropertyUseType')

TARGET = ['SiteEnergyUseWN_kBtu']


def load_data(path):
    """Load raw dataset.

    :args:
        path (path-like object or str) : path to the data that you want load.
    :return:
        data (pd.DataFrame) : loaded data
    """
    return pd.read_pickle(path)


def make_dummies_dataframe(dataframe, columns=CATEGORICAL_FEATURES):
    """Build a dummy (OneHotEncoding) dataframe.

    :args:
        dataframe (pd.DataFrame) : Original dataframe with categories to encode
        columns (list of str) : the categorical features to encode. Must
        contain similar categories.
    :return:
        dummy dataframe : dataframe encoded.

    :usage:
        >>> data = pd.DataFrame({"a": ['foo', 'foo', 'bar'],
                                 "b": ['foo', 'foo', 'foo'],
                                 "c": ['baz', 'foo', 'foo']})
        >>> make_dummies_dataframe(data, columns=['a', 'b', 'c'])
               bar  foo  baz
            0  0.0  1.0  1.0
            1  0.0  1.0  0.0
            2  1.0  1.0  0.0
    """
    dummies = list()
    original_index = dataframe.index
    for categ in columns:
        dummies.append(pd.get_dummies(dataframe[categ]).reset_index(drop=True))
    dataframe = pd.concat(dummies)\
        .groupby(level=0).any().astype('float64')
    dataframe.set_index(original_index, inplace=True)
    return dataframe


def build_features(origin_df, dummies, original_cols=CATEGORICAL_FEATURES):
    """Build the final feature dataset.

    :args:
        origin_df (pd.DataFrame) : Original dataframe.
        dummies (pd.DataFrame) : Dataframe with one hot encoded categories
        original_cols (list of str) : Columns with categorical data in the
        original dataframe.
    :return:
        dummy dataframe with associated GFA values instead of ones.

    :usage:
        >>> data = pd.DataFrame({"usage_1": ['housing', 'housing', 'office'],
                                 "usage_1GFA": [100, 200, 300],
                                 "usage_2": ['parking', 'pool', 'parking'],
                                 "usage_2GFA": [101, 201, 301],
                                 "usage_3": [None, 'parking', 'data center'],
                                 "usage_3GFA": [None, 211, 311]})
        >>> dummies = make_dummies_dataframe(data,
                                             columns=['usage_1',
                                                      'usage_2',
                                                      'usage_3'])
        >>> dummies
               housing  office  parking  pool  data center
          0      1.0     0.0      1.0   0.0          0.0
          1      1.0     0.0      1.0   1.0          0.0
          2      0.0     1.0      1.0   0.0          1.0

        >>> build_features(data, dummies, original_cols=['usage_1',
                                                         'usage_2',
                                                         'usage_3'])
                housing  office  parking   pool  data center
            0    100.0     0.0    101.0    0.0          0.0
            1    200.0     0.0    211.0  201.0          0.0
            2      0.0   300.0    301.0    0.0        311.0
    """
    for col in dummies.columns:
        for categ in original_cols:
            idx = origin_df.loc[origin_df[categ] == col].index
            dummies.loc[idx, col] = origin_df.loc[origin_df[categ] ==
                                                  col][categ + 'GFA']
    return dummies


def drop_null_target(dataframe, target=TARGET):
    """Remove all rows containing null values for target.

    :args:
        dataframe (pd.DataFrame) : dataframe with features and target.
        target (str) : the column name of the target.

    :usage:
        >>> data = pd.DataFrame({'a' : [1, 2, 3],
                                 'b' : [4, 5, 6],
                                 'c' : [7, 0, 9]})
        >>> drop_null_target(data, target='c')
           a  b  c
        0  1  4  7
        2  3  6  9

    """
    return dataframe[dataframe[target] != 0]


def save_features(data, path):
    """Save data after transformation.

    :args:
        data (pd.DataFrame) : dataframe to save
        path (path-like or str) : output filepath where you want to save
        the data.
    """
    data.to_pickle(path)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    data = load_data(project_dir.joinpath('data', 'interim',
                                          'full_data.pickle'))
    dummies = make_dummies_dataframe(data)
    features = build_features(data, dummies)
    features = pd.concat([dummies, data[TARGET]], axis=1)
    save_features(features, project_dir.joinpath('data', 'processed',
                                                 'model_data.pickle'))
