"""
Contains the base class for classification models.
"""
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
#from ..externals import six
import six

class BasicModel(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    """
    Abstract class used as interface for the classificator component.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        TODO
        """

    @abstractmethod
    def predict(self, X, sample_weight=None):
        """
        TODO
        """

    @abstractmethod
    def predict_proba(self, X, sample_weight=None):
        """
        TODO
        """

    def get_name(self):
        """
        TODO
        """
        return self.name