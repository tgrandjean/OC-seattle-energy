"""linear regression.

Linear regression model.
"""
from sklearn.linear_model import LinearRegression

from src.models.model import AbstractModel


class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters={}):
        super().__init__(X, y, hyperparameters)
        self.model = LinearRegression()

    def __repr__(self):
        return "LinearRegression"
