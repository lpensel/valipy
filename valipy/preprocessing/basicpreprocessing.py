"""
Contains the base class for data preprocessing.
"""

class BasicPreprocessing(object):
    """
    Abstract class used as interface for the data-preprocessing component.
    """

    def fit(self, X, y=None, z=None):
        """
        Fit the preprocessing tool to the data.

        Given the data X and optionally the target values y and the sample
        weights z, the preprocessing tool is fitted.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray

        Returns
        -------
        None
        """
        raise NotImplementedError("This needs to be implemented.")

    def transform(self, X, y=None, z=None):
        """
        Transform the given data.

        Returns the transformed data X and optionally y and z. The fit method
        has to be called first for a proper transform.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray

        Returns
        -------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray
        """
        raise NotImplementedError("This needs to be implemented.")

    def fit_transform(self, X, y=None, z=None):
        """
        Fits the tool to the data and transforms the data

        Given the data X and optionally the target values y and the sample
        weights z, the preprocessing tool is fitted. Then the transformed data
        X and optionally y and z are returned.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray

        Returns
        -------
        X : numpy.ndarray
        y : numpy.ndarray
        z : numpy.ndarray
        """
        self.fit(X, y, z)
        return self.transform(X, y, z)