"""train_model

train all models and get the best.
"""
import importlib
import os
from pathlib import Path

import numpy as np
from sklearn import metrics

from src.features.data_processing import ProcessingPipeline


MODELS_PATH = Path(__file__).parents[0].joinpath('models')

model_inputs = ['BuildingType',
                'LargestPropertyUseType',
                'LargestPropertyUseTypeGFA',
                'NumberofFloors',
                'ENERGYSTARScore']

model_target = ['SiteEnergyUseWN_kBtu']


class ModelTrainer(object):

    def __init__(self, data, models_path=MODELS_PATH,
                 model_inputs=model_inputs, model_target=model_target):
        self.data = data
        self.models = self._get_models(models_path)
        self.processing_data = ProcessingPipeline(self.data,
                                                  input_=model_inputs,
                                                  target=model_target)
        self.trained_models = list()

    def _get_models(self, models_path):
        models = []
        for f in os.listdir(models_path):
            if not f.startswith('__'):
                module = f'src.models.models.{f[:-3]}'
                models.append(importlib.import_module(module))
        return [x.Model for x in models]

    def train_models(self):
        for model in self.models:
            m = model(self.processing_data.X, self.processing_data.y)
            m.train()
            try:
                m.plot_optimization_results()
            except TypeError as e:
                pass
            self.trained_models.append(m)

    def score_models(self, X, y):
        rmse_list = []
        if not self.trained_models:
            self.train_models()

        for model in self.trained_models:
            y_pred = model.predict(X)
            rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
            rmse_list.append(rmse)
            print(f"RMSE {model} : {rmse}")

        return rmse_list

    def get_best_model(self, X, y):
        best_index = np.argmin(self.score_models(X, y))
        return self.trained_models[best_index]
