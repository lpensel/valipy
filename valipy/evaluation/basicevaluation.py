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
    def eval(self, results, scoring, no_splits, scoring_weight=None):
        """
        TODO
        """

    def evaluate(self, results, scoring, no_splits, scoring_weight=None):
    	res,_ = self.eval(results, scoring, no_splits, scoring_weight)
    	return res