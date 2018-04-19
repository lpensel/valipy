"""
Contains the class for multi model SK grid search.
"""

import numpy as np


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator


from ..model.basicmodel import BasicModel




class MultiModelSearch(object):
    """
    Grid-Search over multiple estimator and parameter tuples.


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
    """
    def __init__(self, data_handler, parameter, scoring=None, n_jobs=1,
                 cv=5, preprocessing=None, verbose=5, random_state=None):
        self.data_handler = data_handler
        self.scoring = scoring
        self._process_parameter(parameter)
        self._process_cv(cv, random_state)
        self.cv = list(self.cv)
        self.results = None
        self.preprocessing = preprocessing
        self.searches = [GridSearchCV(para[0], para[1], scoring=scoring, 
                                      n_jobs=n_jobs, cv=self.cv, 
                                      verbose=verbose, refit=False)
                         for para in self.parameter]

    def _process_parameter(self, parameter):
        if isinstance(parameter[0],BaseEstimator):
            self.parameter = [parameter]
        else:
            self.parameter = parameter

    def _process_cv(self, cv, random_state):
        if isinstance(cv, int):
            X,y,_ = self.data_handler.get_train()
            skf = StratifiedKFold(n_splits=cv, random_state=random_state)
            self.cv = skf.split(X,y)
        else:
            self.cv = cv

    def _update_result(self,results,estimator):
        if isinstance(estimator, BasicModel):
            name = estimator.get_name()
        else:
            name = estimator.__class__.__name__

        for i in range(len(results["params"])):
            results["params"][i]["Estimator"] = name
        if self.results is None:
            self.results = results
        else:
            for key in results.keys():
                if "param_" not in key:
                    self.results[key] = np.concatenate((self.results[key],
                                                        results[key]))

    def fit(self):
        """
        Running the Grid-Search for all estimators and all parameters.
        """
        X, y, z = self.data_handler.get_train()
        if self.preprocessing is not None:
            X, y, z = self.preprocessing.fit_transform(X, y, z)
        for i in range(len(self.searches)):
            print("MODEL %d" % (i))
            search = self.searches[i]
            search.fit(X,y)
            self._update_result(search.cv_results_, self.parameter[i][0])

        




























