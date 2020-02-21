from pathlib import Path
import pandas as pd

CATEGORICAL_FEATURES = ('LargestPropertyUseType',
                        'SecondLargestPropertyUseType',
                        'ThirdLargestPropertyUseType')

TARGET = ['SiteEnergyUseWN_kBtu']


def load_data(path):
    """Load raw dataset."""
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
    return dataframe[dataframe[target] != 0]


def save_features(data, path):
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
