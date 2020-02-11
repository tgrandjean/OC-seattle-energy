"""Data processing.

Collection of functions/methods to process data for modeling.
"""
from abc import ABC
import logging

import pandas as pd

logger = logging.getLogger(__name__)

class AbstractPipeline(ABC):
    """AbstractPipeline,

    Parent class for data pipeline.

    """

    def __init__(self, raw_data):
        logger.debug("Initializing DataPipeline")
        self.data = raw_data
        logger.debug("Initialized DataPipeline")

    @property
    def data(self):
        """Get or set the current data set.
        Data should be a pandas DafaFrame object.
        """
        return self._data

    @data.setter
    def data(self, data):
        if type(data) != pd.DataFrame:
            raise ValueError("You should pass a pandas dataframe obj.")

    def process(self):
        """process the pipeline.

        Top level api method.
        Execute this method to run the pipeline.
        """
        raise NotImplementedError("Override this method.")
