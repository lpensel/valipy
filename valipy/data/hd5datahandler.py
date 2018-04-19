"""
Contains the data handler for hd5 files used in the thesis.
"""
import h5py
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from .basicdatahandler import BasicDataHandler

class HD5Data(BasicDataHandler):
    """
    TODO
    """

    def __init__(self, data_path, attributes, labels, multi_class=False):
        """
        TODO
        """
        self.data_path = data_path
        self.attributes = attributes
        self.labels = labels
        self.target = labels[0]
        self.multi_class = multi_class
        self._branch_names = self._get_branch_names()
        self._filter_attributes()
        self._filter_samples()

    def _get_branch_names(self):
        """
        TODO
        """
        with h5py.File(self.data_path, "r") as h5f:
            branches=h5f.get("branch_names")
            branch_names = []
            for i in range(branches.shape[0]):
                if len(branches.shape)>1:
                    branch_names.append(branches[i,0].decode("unicode_escape"))
                else:
                    branch_names.append(branches[i].decode("unicode_escape"))
        return branch_names

    def _filter_attributes(self):
        """
        TODO
        """
        attribute_ids = []
        attribute_names = []
        for b in range(len(self._branch_names)):
            if self._branch_names[b] in self.attributes:
                attribute_ids.append(b)
                attribute_names.append(self._branch_names[b])
        #Error detection
        if len(attribute_ids) is not len(self.attributes):
            error_msg = "Attributes "
            for name in self.attributes:
                if name not in attribute_names:
                    error_msg += name + " "
            error_msg += "not found."
            raise ValueError(error_msg)
        #Read data
        with h5py.File(self.data_path, "r") as h5f:
            self.X_test=np.array(h5f.get("X_test_data")[:,attribute_ids])
            self.X_train=np.array(h5f.get("X_train_data")[:,attribute_ids])
            self.y_test=np.array(h5f.get("y_test_data"))
            self.y_train=np.array(h5f.get("y_train_data"))
            self.z_test=np.array(h5f.get("z_test_data"))
            self.z_train=np.array(h5f.get("z_train_data"))

    def _filter_samples(self):
        """
        TODO
        """
        #Training data
        sample_ids = []
        for d in range(len(self.y_train)):
            if self.y_train[d] in self.labels:
                sample_ids.append(d)
                if not self.multi_class:
                    if self.y_train[d] == self.target:
                        self.y_train[d] = 1
                    else:
                        self.y_train[d] = 0
        self.X_train = self.X_train[sample_ids,:]
        self.y_train = self.y_train[sample_ids]
        self.z_train = self.z_train[sample_ids]
        if self.multi_class:
            enc = OneHotEncoder()
            self.y_train = enc.fit_transform(
                           self.y_train[:,np.newaxis]).toarray()
        #Testing data
        sample_ids = []
        for d in range(len(self.y_test)):
            if self.y_test[d] in self.labels:
                sample_ids.append(d)
                if not self.multi_class:
                    if self.y_test[d] == self.target:
                        self.y_test[d] = 1
                    else:
                        self.y_test[d] = 0
        self.X_test = self.X_test[sample_ids,:]
        self.y_test = self.y_test[sample_ids]
        self.z_test = self.z_test[sample_ids]
        if self.multi_class:
            enc = OneHotEncoder()
            self.y_test = enc.fit_transform(
                           self.y_test[:,np.newaxis]).toarray()


























