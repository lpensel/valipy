"""
Contains the class for multi model SaveGridSearch search.
"""


from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

from . import SaveGridSearchCV
from . import MultiModelSearch




class MultiModelSaveSearch(MultiModelSearch):
    """
    Save-Grid-Search over multiple estimator and parameter tuples.


    Parameters:
    -----------
    data_handler:   BasicDataHandler
                    Used as data source for the search.
    parameter:      tuple or list of tuples
                    Has the form of (estimator, parameters) or a list of those
                    tuples.
    scoring:        string, callable, list/tuple, dict or None, default: None
                    Metrics used for evaluating the search results
    cv:             int, cross-validation generator or an iterable, default: 5
                    If number it defines the number of splits.
                    Otherwise it gives an cv generator or a list of splits.
    preprocessing:  BasicPreprocessing or None, default: None
                    If the data should be preprocessed.
    verbose:        integer, optional
                    High number give more output.
    save:           None means no saving
                    String used as prefix for result directories
    """
    def __init__(self, data_handler, parameter, scoring=None, n_jobs=1, cv=5,
                 preprocessing=None, verbose=5, random_state=None, save=None):
        self.data_handler = data_handler
        self.scoring = scoring
        self._process_parameter(parameter)
        self._process_cv(cv, random_state)
        self.cv = list(self.cv)
        self.results = None
        self.preprocessing = preprocessing
        self.fitted = False
        self.evaluated = False
        self.model_build = False
        self.searches = [SaveGridSearchCV(para[0], para[1], save=save, 
                                   scoring=scoring, n_jobs=n_jobs, cv=self.cv, 
                                   verbose=verbose, refit=False)
                         for para in self.parameter]

    
























