"""univar

univariate analysis.

Thibault Grandjean
"""

import warnings

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

    def make_analysis(self, column):
        """Make full analysis."""
        print(self.completion_rate(column))
        if self.data[column].dtype.name in ['int64', 'float64']:
            tab = tabulate(self.series_stats(column), tablefmt="html")
            display(HTML(tab))
            self.graph_series(column)
        else:
            raise NotImplementedError('This feature will be available soon.')

    def outliers_infos(self, column, lower_trig=0.05, upper_trig=0.95):
        """Return informations for values out of range of quantiles."""
        lower = self.data[self.data[column] <
                          self.data[column].quantile(lower_trig)].copy()
        upper = self.data[self.data[column] >
                          self.data[column].quantile(upper_trig)].copy()
        return lower, upper
