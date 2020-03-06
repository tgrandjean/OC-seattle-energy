"""univar

univariate analysis.

Thibault Grandjean
"""

import warnings

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.stattools import medcouple
from tabulate import tabulate


class UnivariateAnalysis(object):
    """UnivariateAnalysis."""

    def __init__(self, dataframe):
        plt.rcdefaults()
        font = {'size': 16}
        sns.set()
        plt.rc('font', **font)
        self.data = dataframe

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dataframe):
        if type(dataframe) != pd.DataFrame:
            raise ValueError("You must pass a pandas's dataframe.")
        self._data = dataframe

    def get_missing(self, column):
        """Return a dataframe with missing value in the `column`."""
        return self.data[self.data[column].isna()]

    def completion_rate(self, column):
        """return completion rate for a column."""
        complete = self.data[column].dropna().shape[0] \
            / self.data[column].shape[0] * 100
        return f"completion rate -- {column} : {complete} %"

    def series_stats(self, column):
        """Return stats for a Series."""
        series = self.data[column]
        return [['mean', 'std',
                 'min', 'max',
                 'median', 'variance',
                 '25%', '75%'],
                [series.mean(), series.std(),
                 series.min(), series.max(),
                 series.median(), series.var(),
                 series.quantile(0.25), series.quantile(0.75)]]

    def graph_series(self, column):
        """Make a plot for a series."""
        if self.data[column].count() != self.data[column].shape[0]:
            warnings.warn('NaN detected in the series.'
                          ' NaNs are not considered for calculation.')

        # Cut the window in 2 parts
        kwrgs = {"height_ratios": (.15, .85)}
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(8, 8),
                                            gridspec_kw=kwrgs)

        # Add a graph in each part
        sns.boxplot(self.data[column].dropna(), ax=ax_box)
        sns.distplot(self.data[column].dropna(), ax=ax_hist)

        # Remove x axis name for the boxplot
        ax_box.set(xlabel='')
        plt.show()

    def make_analysis(self, column, **kwargs):
        """Make full analysis."""
        print(self.completion_rate(column))
        if self.data[column].dtype.name in ['int64', 'float64']:
            tab = tabulate(self.series_stats(column), tablefmt="html")
            display(HTML(tab))
            self.graph_series(column)
            display(self.outliers(column))
            self.data = self.data_without_outliers(column)
            self.graph_series(column)

        elif self.data[column].dtype.name == 'category':
            self.categorical_analysis(column=column, **kwargs)
        else:
            raise NotImplementedError('This feature will be available soon.')

    def outliers(self, column):
        """Return outliers.

        make correction for skewed data. see :
        Mia Hubert & al 2007 - Outlier detection for skewed data
        """
        quartile_1 = self.data[column].quantile(0.25)
        quartile_3 = self.data[column].quantile(0.75)
        iqr = quartile_3 - quartile_1
        mc = medcouple(self.data[column])
        if mc > 0:
            lower_bound = quartile_1 - (iqr * 1.5 * np.exp(-4 * mc))
            upper_bound = quartile_3 + (iqr * 1.5 * np.exp(3 * mc))
        else:
            lower_bound = quartile_1 - (iqr * 1.5 * np.exp(-3 * mc))
            upper_bound = quartile_3 + (iqr * 1.5 * np.exp(4 * mc))
        return self.data[(self.data[column] < lower_bound) |
                         (self.data[column] > upper_bound)]

    def data_without_outliers(self, column):
        """Return data without outliers."""
        return self.data.drop(self.outliers(column).index)

    def categorical_analysis(self, column, **kwargs):
        """Perform statistical analysis on categorical variable."""
        if kwargs.get('orient') == 'h':
            kwargs['y'] = column
        else:
            kwargs['x'] = column
        fig, ax = plt.subplots(1, figsize=kwargs.pop('figsize', (8, 8)))
        sns.countplot(data=self.data, ax=ax, **kwargs)
