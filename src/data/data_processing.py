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
        self._data = data

    def _column_validator(self, column):
        """Assert that a column or list of columns are in self.data

        :args:
            column (str or list of str) : string or list of string to validate
        """
        if type(column) != str and type(column) != list:
            raise ValueError('Incorrect type must be a string'
                             ' or list of string')
        elif type(column) == str:
            if column not in self.data.columns:
                raise IndexError("Invalid column name %s" % column)
        elif type(column) == list:
            for elt in column:
                if elt not in self.data.columns:
                    raise IndexError('Invalid target name %s' % elt)
        return column

    @property
    def target(self):
        """Get or set target variable **y** of the model.

        target (str or list): string or list of target variables.

        :note:
            must be present in self.data.columns
        """
        return self._target

    @target.setter
    def target(self, target):
        self._target = self._column_validator(target)

    @property
    def input(self):
        """Get or set input variable **X** of the model.

        input (str or list): string or list of input variables.

        :note:
            must be present in self.data.columns
        """
        return self._input

    @input.setter
    def input(self, input):
        self._input = self._column_validator(input)

    def process(self):
        """process the pipeline.

        Top level api method.
        Execute this method to run the pipeline.
        """
        raise NotImplementedError("Override this method.")


class ProcessingPipeline(AbstractPipeline):
    """Data processing pipeline.

    :usage:
        >> data = pd.DataFrame({'x': np.arange(10),
        >>                      'y': np.arange(10)})
        >> pipeline = ProcessingPipeline(data)
    """

    def process(self):
        pass

    def
