"""bivar

bivariate analysis.

Thibault Grandjean
"""
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.features.univar import UnivariateAnalysis


class BivariateAnalysis(UnivariateAnalysis):
    """BivariateAnalysis."""

    def get_correlation(self, variables, **kwargs):
        corr = self.data[variables].corr(**kwargs)
        display(corr)
        return corr

    def anova(self, outcome_variable, group, **kwargs):
        self.boxplot(x=group, y=outcome_variable, **kwargs)
        results = ols(f'{outcome_variable} ~ C({group})', data=self.data).fit()
        aov_table = sm.stats.anova_lm(results, typ=2)
        display(results.summary())
        display(aov_table)
        return results, aov_table

    def boxplot(self, x, y, figsize=(12, 8),
                label_rotation=45, orient='v'):
        fig, ax = plt.subplots(1, figsize=figsize)
        if orient == 'h':
            x, y = y, x
        _ = sns.catplot(x=x,
                        y=y,
                        orient=orient,
                        kind="box",
                        data=self.data, ax=ax)
        plt.close(2)
        plt.xticks(rotation=label_rotation)
        plt.show()

    def scatterplot(self, variables):
        sns.jointplot(x=variables[0],
                      y=variables[1],
                      data=self.data)

    def make_analysis(self, variables):
        """Make full analysis."""
        assert len(variables) == 2
