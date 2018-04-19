"""
Contains the generic data handler.
"""

from .basicdatahandler import BasicDataHandler

class DataHandler(BasicDataHandler):
    """
    TODO
    """

    def __init__(self, train, test):
        """
        TODO
        """
        if len(train) == 3:
            self.X_train = train[0]
            self.y_train = train[1]
            self.z_train = train[2]
        elif len(train) == 2:
            self.X_train = train[0]
            self.y_train = train[1]
            self.z_train = None
        else:
            raise ValueError("Insufficient training data")

        if len(test) == 3:
            self.X_test = test[0]
            self.y_test = test[1]
            self.z_test = test[2]
        elif len(test) == 2:
            self.X_test = test[0]
            self.y_test = test[1]
            self.z_test = None
        else:
            raise ValueError("Insufficient testing data")


























