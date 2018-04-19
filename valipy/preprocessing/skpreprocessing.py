"""
Contains wrapper for scikit-learn preprocessing tools.
"""

from .basicpreprocessing import BasicPreprocessing

class SKPreprocessing(BasicPreprocessing):
    """
    TODO
    """

    def __init__(self,processor):
        self.processor = processor

    def fit(self, X, y=None, z=None):
        """
        TODO
        """
        self.processor.fit(X)

    def transform(self, X, y=None, z=None):
        """
        TODO
        """
        return self.processor.transform(X), y, z
















