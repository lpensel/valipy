import os,sys,inspect

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


#Import not installed ValiPy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(
                             inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import valipy as vp


#set data_handler properties: location of file
parent = os.path.abspath('..')
data_path = str(parent) + "/data/iris.h5"
#which attributes should be used
attributes = np.array(["Sepal length","Sepal width","Petal length",
                       "Petal width"])
#which classes should be used, first decides the target for 1 vs all
labels = np.array([0,1,2])

#build data_handler, multi_class not supported for grid search
data_handler = vp.data.HD5Data(data_path,attributes,labels,multi_class=False)

#The used algorithms, scikit-learn classes can be directly incorporated
keras_mlp = vp.model.KerasMLP()
sk_mlp = MLPClassifier()
skmodel_mlp = vp.model.SKModel(MLPClassifier())

#Parameter sets to test
param_keras_mlp = {"hidden_layer_sizes":[(25,),(50,)]}
param_sk_mlp = {"hidden_layer_sizes":[(25,),(50,)]}
param_skmodel_mlp = {"estimator__hidden_layer_sizes":[(25,),(50,)]}

#combined for multimodel search
parameter=[(keras_mlp,param_keras_mlp),(sk_mlp,param_sk_mlp),
           (skmodel_mlp,param_skmodel_mlp)]

#scoring metrics
scoring = {"AUC": "roc_auc", "Accuracy": "accuracy", 
           "Precision": "average_precision"}

#use standard scaler as preprocessing tool
preprocessing = vp.preprocessing.SKPreprocessing(StandardScaler())

#Prefix for "temp" directories. Do not use test as prefix in this script!!!
pre_dir = "temp"

#do grid search over multiple algorithms
search = vp.search.MultiModelSaveSearch(data_handler, parameter, scoring=scoring,
                 cv=2, preprocessing=preprocessing, save=pre_dir)
search.fit()
ranking = search.evaluate(scoring_weight=[2.0,1.0,2.0])

#print sorted results
for a,b in ranking:
    print("Parameter: {}\nScore: {}".format(a,b))


estimator = search.get_model()
X, y, z = data_handler.get_test()
X, y, z = search.preprocess(X,y,z)

print("Predicting Test set")
y_pred = estimator.predict(X)
y_prob = estimator.predict_proba(X)
y_prob = y_prob[:,1]

for i in range(len(y_pred)):
	print("{}.: true = {}, pred = {}, proba = {}".format(i,y[i],y_pred[i],
														 y_prob[i]))

#Delete "temp" directories(for UNIX)
os.system("rm -r {}_*".format(pre_dir))