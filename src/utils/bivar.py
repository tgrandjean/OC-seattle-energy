"""bivar

bivariate analysis.

Thibault Grandjean
"""
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison

from statsmodels.formula.api import ols
from scipy import stats
from tabulate import tabulate

from src.utils.univar import UnivariateAnalysis


def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


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
        self.pairwise_hsd(outcome_variable, group, **kwargs)
        return results, aov_table

    def pairwise_hsd(self, outcome_variable, group, **kwargs):
        cleandata = self.data.filter(items=[outcome_variable,
                                            group]).dropna()
        mc1 = MultiComparison(cleandata[outcome_variable],
                              cleandata[group])
        results = mc1.tukeyhsd()
        results.plot_simultaneous(figsize=kwargs.get('figsize', (12, 8)))
        display(results.summary())

    def boxplot(self, x, y, figsize=(12, 8),
                label_rotation=45, orient='v'):
        fig, ax = plt.subplots(1, figsize=figsize)
        if orient == 'h':
            x, y = y, x
        _ = sns.boxplot(x=x,
                        y=y,
                        orient=orient,
                        data=self.data, ax=ax)
        plt.close(2)
        plt.xticks(rotation=label_rotation)
        plt.show()

    def scatterplot(self, variables):
        sns.jointplot(x=variables[0],
                      y=variables[1],
                      data=self.data)

    # def make_analysis(self, variables):
    #     """Make full analysis."""
    #     assert len(variables) == 2
    def regression(self, variables):
        """Create a regression."""
        sns.regplot(x=variables[0], y=variables[1], data=self.data)
        results = ols(f'{variables[0]} ~ {variables[1]}',
                      data=self.data).fit()
        display(results.summary())
        return results

    def chi_square_contingency(self, variables):
        """Make a Chi square analysis between two categorical variables."""
        cont = pd.crosstab(self.data[variables[0]],
                           self.data[variables[1]])
        print('Contingency table')
        cm = sns.light_palette("green", as_cmap=True)

        display(cont.style.apply(background_gradient,
                                 cmap=cm,
                                 m=cont.min().min(),
                                 M=cont.max().max(),
                                 low=0,
                                 high=0.2))
        chi2_stat, p_val, dof, ex = stats.chi2_contingency(cont)
        tab = tabulate([['Chi_2 stat', chi2_stat],
                        ['P value', p_val],
                        ['dof', dof]
                        ], tablefmt='html')
        display(HTML(tab))
        print('='*30)
        print('Expected fequencies')
        cont.iloc[:, :] = ex

        display(cont.style.apply(background_gradient,
                                 cmap=cm,
                                 m=cont.min().min(),
                                 M=cont.max().max(),
                                 low=0,
                                 high=0.2))
        print(ex.shape)
