"""lasso.

Lasso regression model.
"""
from sklearn.linear_model import Lasso

from src.models.model import AbstractModel


class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters=dict(alpha=0.1)):
        super().__init__(X, y, hyperparameters)
        self.model = Lasso(alpha=self.hyperparameters.get('alpha'))
