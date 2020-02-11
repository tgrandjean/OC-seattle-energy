"""Data processing.

Collection of functions/methods to process data for modeling.
"""
from abc import ABC
import logging

import pandas as pd
from sklearn import preprocessing

logger = logging.getLogger(__name__)

class AbstractPipeline(ABC):
    """AbstractPipeline,

    Parent class for data pipeline.
    """

    def __init__(self, raw_data, input_, target):
        logger.debug("Initializing DataPipeline")
        self.data = raw_data
        self.input = input_
        self.target = target
        logger.debug("Initialized DataPipeline")

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
                column = [column]
        elif type(column) == list:
            for elt in column:
                if elt not in self.data.columns:
                    raise IndexError('Invalid target name %s' % elt)
        return column  # type(column) == list

    @property
    def data(self):
        """Get or set the current data set.
        Data must be a pandas DafaFrame object.
        """
        return self._data

    @data.setter
    def data(self, data):
        if type(data) != pd.DataFrame:
            raise ValueError("You should pass a pandas dataframe obj.")
        self._data = data

    @property
    def target(self):
        """Get or set target variable **y** of the model.

        target (str or list): string or list of target variables converted
        in list (if string, that will give us a one item list.).

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
    def input(self, input_):
        self._input = self._column_validator(input_)

    @property
    def y(self):
        """Get y (canonical target data for modeling)."""
        return self._y.values

    @property
    def X(self):
        """Get X (canonical input data for modeling)."""
        return self._X.values

    @X.setter
    def X(self, dataframe):
        assert set(list(dataframe.columns.values)) == set(self.input)
        self._X = dataframe

    @y.setter
    def y(self, dataframe):
        assert set(list(dataframe.columns.values)) == set(self.target)
        self._y = dataframe

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
        self.handle_missing_values()
        self.scale_numerical_data()
        self.X = self._scaled_data[self.input]
        self.y = self._scaled_data[self.target]
        self.encode_categorical_data()

    def scale_numerical_data(self):
        """scale numerical data for modeling

        :note:
            see the scikit doc:
            https://scikit-learn.org/stable/auto_examples/preprocessing/
            plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-
            plot-all-scaling-py

        """
        logger.info("Scaling numerical data between -1 and 1.")
        self._scaled_data = self.data.copy()
        self._scalers = dict()
        for col in self._scaled_data.columns:
            if col not in self.input and col not in self.target:
                self._scaled_data.drop(col, axis=1, inplace=True)

        for col in self._scaled_data.columns:
            logger.debug("scaling %s", col)
            if self.data[col].dtype in [float, int]:
                x = self._scaled_data[col].values.reshape(-1, 1)
                self._scalers[col] = preprocessing.StandardScaler().fit(x)
                self._scaled_data.loc[:, col] = self._scalers[col].transform(x)
                min_ = self._scaled_data[col].min()
                max_ = self._scaled_data[col].max()
            else:
                print(f"ignoring {col}, dtype {self.data[col].dtype.name}")

    def encode_categorical_data(self):
        for col in self._X.columns:
            if self._X[col].dtype.name == 'category':
                self._X = pd.concat([self._X,
                                     pd.get_dummies(self._X[col],
                                                    prefix=col,
                                                    dummy_na=True)],
                                        axis=1)
                self._X.drop(col, axis=1, inplace=True)

    def handle_missing_values(self):
        """Handle missing value :

        missing observations in target are droped and missing values
        in input are imputed to zero.
        """
        for col in self.target:
            print(col)
            self.data = self.data[self.data[col].notnull()]
        for col in self.input:
            print(col)
            if self.data[col].dtype.name == "category":
                print(f"ignoring {col}: categorical data")
            else:
                self.data.loc[self.data[col].isnull(), col] = 0

    @property
    def scaled_data(self):
        return self._scaled_data

    @property
    def scalers(self):
        return self._scalers
