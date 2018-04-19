"""
Contains the base class for data handling.
"""

import random
import numpy as np

class BasicDataHandler(object):
    """
    Abstract class used as interface for the data-handling component.

    Parameters
    ----------
    data_path : str

    Attributes
    ----------
    data_path : str
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.z_train = None
        self.X_test = None
        self.y_test = None
        self.z_test = None

    def get_train(self):
        """
        Provides the training-data for the contained data source.

        The data should be returned as (X, y, z), where X
        contains the attribute values for each sample, y contains the 
        target values and z are optional weights for each sample.

        Parameters
        ----------
        None

        Returns
        -------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray
        """
        X = self.X_train
        y = self.y_train
        z = self.z_train
        return X, y, z

    def get_test(self):
        """
        Provides the testing-data for the contained data source.

        The data should be returned as (X, y, z), where X
        contains the attribute values for each sample, y contains the 
        target values and z are optional weights for each sample.

        Parameters
        ----------
        None

        Returns
        -------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray
        """
        X = self.X_test
        y = self.y_test
        z = self.z_test
        return X, y, z






