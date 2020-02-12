"""model module.

Define here an abstract class for model.
"""
from abc import ABC


class AbstractModel(ABC):

    def __init__(self, X, y, hyperparameters={}):
        self.X = X
        self.y = y
        self.model = None
        self.hyperparameters = hyperparameters

    def __repr__(self):
        return f"{self.model}"

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters
