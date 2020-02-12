"""regression ridge.

Ridge regression model.
"""
from sklearn.linear_model import Ridge

from src.models.model import AbstractModel


class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters=dict(alpha=1.0)):
        super().__init__(X, y, hyperparameters)
        self.model = Ridge(alpha=self.hyperparameters.get('alpha'))
