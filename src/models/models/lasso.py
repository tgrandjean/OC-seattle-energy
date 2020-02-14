"""lasso.

Lasso regression model.
"""
import numpy as np
from sklearn.linear_model import Lasso

from src.models.model import AbstractModel

ALPHAS = alpha=np.logspace(3, 6, num=100)


class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters=dict(alpha=ALPHAS)):
        super().__init__(X, y, hyperparameters)
        self.model = Lasso(alpha=self.hyperparameters.get('alpha'))

    def __repr__(self):
        return 'Lasso'
