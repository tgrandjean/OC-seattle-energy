"""elastic_net.

Elastic Net regression model.
"""
import numpy as np
from sklearn.linear_model import ElasticNet

from src.models.model import AbstractModel

ALPHAS = alpha=np.logspace(-2, 2, num=100)


class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters=dict(alpha=ALPHAS)):
        super().__init__(X, y, hyperparameters)
        self.model = ElasticNet()

    def __repr__(self):
        return "ElasticNet"
