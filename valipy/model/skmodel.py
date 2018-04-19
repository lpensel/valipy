"""
Contains the class used to incorporate sklearn models into the system.
"""

from .basicmodel import BasicModel

class SKModel(BasicModel):
    """
    TODO
    """

    def __init__(self, estimator=None, name=None):

        self.estimator = estimator
        if name is None:
            name = "SKModel_{}".format(estimator.__class__.__name__)
        super(SKModel, self).__init__(name)
        

    def fit(self, X, y, sample_weight=None):
        """
        TODO
        """
        if sample_weight is None:
            self.estimator.fit(X,y)
        else:
            self.estimator.fit(X,y,sample_weight)

    def predict(self, X, sample_weight=None):
        """
        TODO
        """

        return self.estimator.predict(X)

    def predict_proba(self, X, sample_weight=None):
        """
        TODO
        """
        return self.estimator.predict_proba(X)
