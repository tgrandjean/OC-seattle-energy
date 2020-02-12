"""elastic_net.

Elastic Net regression model.
"""
from sklearn.linear_model import ElasticNet

from src.models.model import AbstractModel

class Model(AbstractModel):

    def __init__(self, X, y, hyperparameters={}):
        super().__init__(X, y, hyperparameters)
        self.model = ElasticNet()
