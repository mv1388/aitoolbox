import numpy as np
from pandas import DataFrame

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


class RRegressionModel:
    def __init__(self, X, y, X_names=None, y_names=None):
        """

        Args:
            X:
            y:
            X_names:
            y_names:
        """
        self.X = np.array(X)
        self.y = np.array(y)

        if X_names is None:
            X_names = ["Attr_" + str(idx) for idx in range(0, self.X.shape[1])]
            y_names = ["y_attr"]

        self.X_names = X_names
        self.y_names = y_names
        self.data_set_names = self.X_names + self.y_names

        self.data_frame_pandas = DataFrame(np.column_stack((self.X, self.y)))
        self.data_frame_pandas.columns = self.data_set_names

        pandas2ri.activate()
        self.R_data_frame = pandas2ri.py2ri(self.data_frame_pandas)
        pandas2ri.deactivate()

        self.fit = None
        self.stats = importr('stats')

    def poisson_regression_fit(self, formula, regression_family):
        """

        Args:
            formula:
            regression_family:

        Returns:

        """
        self.fit = self.stats.glm(formula, data=self.R_data_frame, family=regression_family)
        return self

    def predict(self, X_new):
        """

        Args:
            X_new:

        Returns:

        """
        data_frame_pandas_new = DataFrame(X_new)
        data_frame_pandas_new.columns = self.X_names

        pandas2ri.activate()
        R_data_frame_new = pandas2ri.py2ri(data_frame_pandas_new)

        prediction = self.stats.predict(self.fit, R_data_frame_new)

        return prediction

    def print_summary(self):
        """

        Returns:

        """
        base = importr('base')
        print(base.summary(self.fit))
