import numpy as np
import h5py
import os

from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit

parent = os.path.abspath('..')
data_path = str(parent) + "/data/"

iris = datasets.load_iris()
X = iris.data
y = iris.target
z = np.ones(y.shape[0])

attributes = np.array(["Sepal length","Sepal width","Petal length",
                       "Petal width"], dtype='S')
classnames = np.array(["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"], 
                      dtype='S')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_index, test_index = list(sss.split(X, y))[0]

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
z_train, z_test = z[train_index], z[test_index]

with h5py.File(data_path + "iris.h5", "w") as f:
    dset_X_test = f.create_dataset("X_test_data", data=X_test)
    dset_X_train = f.create_dataset("X_train_data", data=X_train)
    dset_attributes = f.create_dataset("branch_names", data=attributes)
    dset_classnames = f.create_dataset("classnames", data=classnames)
    dset_y_test = f.create_dataset("y_test_data", data=y_test)
    dset_y_train = f.create_dataset("y_train_data", data=y_train)
    dset_z_test = f.create_dataset("z_test_data", data=z_test)
    dset_z_train = f.create_dataset("z_train_data", data=z_train)