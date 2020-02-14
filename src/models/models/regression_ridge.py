"""regression ridge.

Ridge regression model.
"""
import numpy as np
from sklearn.linear_model import Ridge

from src.models.model import AbstractModel

ALPHAS = alpha=np.logspace(-1, 3, num=100)


class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters=dict(alpha=ALPHAS)):
        super().__init__(X, y, hyperparameters)
        self.model = Ridge(alpha=self.hyperparameters.get('alpha'))

    def __repr__(self):
        return "Ridge"
