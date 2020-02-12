"""model module.

Define here an abstract class for model.
"""
from abc import ABC

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

sns.set()


class AbstractModel(ABC):

    def __init__(self, X, y, hyperparameters={}):
        self.X = X
        self.y = y
        self.model = None
        self.hyperparameters = hyperparameters

    def __repr__(self):
        return f"{self.model}"

    def train(self):
        print(f"training {self.__repr__()}")
        if self.hyperparameters:
            print(f"optimize {list(self.hyperparameters.keys())}")
            clf = GridSearchCV(self.model, self.hyperparameters,
                               scoring='neg_root_mean_squared_error',
                               n_jobs=-1)
            clf.fit(self.X, self.y)
            self._clf = clf
            self.model = clf.best_estimator_.set_params(**clf.best_params_)
        else:
            self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def _optimization_results(self):
        try:
            return self._clf.cv_results_
        except AttributeError as e:
            if not self.hyperparameters:
                print("No hyperparameters to optimize")
            else:
                raise e

    def plot_optimization_results(self):
        for param in self.hyperparameters.keys():
            plt.plot(self._optimization_results[f'param_{param}'],
                     - self._optimization_results['mean_test_score'])
            ax = plt.gca()
            ax.set_xscale('log')
            plt.xlabel(param)
            plt.ylabel("RMSE")
            plt.show()

    def plot_coeff_as_function(self):
        coeffs = list()
        for param, values in self.hyperparameters.items():
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 12))
            for val in values:
                self.model.set_params(**{param: val})
                self.model.fit(self.X, self.y)
                coef_ = self.model.coef_
                if coef_.shape != (self.X[0].shape):
                    coef_ = coef_.reshape(self.X[0].shape, )
                coeffs.append(coef_)
            self.model = self.model.set_params(**self._clf.best_params_)
            print(self.model.coef_.shape)
            ax[0].plot(values, coeffs)
            ax[0].set_xscale('log')
            ax[1].plot(values, - self._optimization_results['mean_test_score'])
            ax[1].set_xscale('log')
            ax[0].vlines(self.model.get_params()[param], *ax[0].get_ylim(),
                         linestyles='dashed', colors=['b'])
            ax[1].vlines(self.model.get_params()[param], *ax[1].get_ylim(),
                         linestyles='dashed', colors=['b'])
            ax[1].hlines(- self._clf.best_score_, 0, ax[1].get_xlim()[1],
                         linestyles='dashed', colors=['b'])
            ax[1].set_ylabel('RMSE')
            ax[0].set_ylabel('Coeff')
            plt.xlabel(param)
            plt.show()
