"""
Contains the base class for evaluations.
"""
from abc import ABCMeta, abstractmethod
#from ..externals import six
import six

class BasicEvaluation(six.with_metaclass(ABCMeta)):
    """
    Abstract class used as interface for the classificator component.
    """

    @abstractmethod
    def evaluate(self, results, scoring, cv, scoring_weight=None):
        """
        TODO
        """