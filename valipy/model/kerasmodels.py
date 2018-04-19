"""
Contains the class used to incorporate Keras MLP models into the system.
"""
import numpy as np

import six

from abc import ABCMeta, abstractmethod

from keras.models import Sequential
from keras.layers import Dense, LocallyConnected1D, Conv1D, Flatten
from keras import regularizers
from keras import initializers
from keras.callbacks import EarlyStopping

from .basicmodel import BasicModel



def _preprocess_vars(variable, n):
    if type(variable) == type(()) or type(variable) == type([]):
        if len(variable) == n:
            return variable
        return tuple(variable[0] for i in range(n))
    return tuple(variable for i in range(n))





class KerasBase(six.with_metaclass(ABCMeta, BasicModel)):
    """
    TODO
    """
    def __init__(self, name, max_iter, batch_size, verbose, validation_fraction,
                 shuffle, early_stopping, tol, patience, is_convolution=False):
        super(KerasBase, self).__init__(name)
        self.is_convolution = is_convolution
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self._n_attributes = None
        self._n_classes = None
        self._model = None
        self._callbacks = []
        if early_stopping:
            monitor = "loss"
            if self.validation_fraction > 0.0:
                monitor = "val_loss"
            self._callbacks.append(EarlyStopping(monitor=monitor, min_delta=tol, 
                                                 patience=patience, 
                                                 verbose=self.verbose, 
                                                 mode="auto"))

    def fit(self, X, y, sample_weight=None):
        """
        TODO
        """
        _,self._n_attributes = X.shape
        if self.is_convolution:
            X = np.expand_dims(X, axis=2)
        if len(y.shape) == 1:
            self._n_classes = 1
        else:
            _,self._n_classes = y.shape
        self._build_model()
        self._model.fit(X, y, epochs=self.max_iter, 
                        batch_size=self.batch_size, verbose=self.verbose, 
                        callbacks=self._callbacks, 
                        validation_split=self.validation_fraction,
                        shuffle = self.shuffle, 
                        sample_weight=sample_weight)

    def predict(self, X, sample_weight=None):
        """
        TODO
        """

        pred = self.predict_proba(X,sample_weight)
        return pred.argmax(axis=-1)

    def predict_proba(self, X, sample_weight=None):
        """
        TODO
        """
        if self.is_convolution:
            X = np.expand_dims(X, axis=2)
        pred = self._model.predict(X, batch_size=self.batch_size, 
                                   verbose=self.verbose)
        if self._n_classes == 1:
            return np.concatenate((1-pred,pred), axis=1)
        else:
            return pred

    @abstractmethod
    def _build_model(self):
        """
        TODO
        """






class KerasMLP(KerasBase):
    """
    TODO
    """


    def __init__(self, hidden_layer_sizes=(100, ), activation="relu", 
                 solver="adam", l2_kernel=0.0, l2_activation=0.0, 
                 batch_size=200, max_iter=50, shuffle=True, random_state=None, 
                 tol=0.0001, verbose=2, early_stopping=True, 
                 validation_fraction=0.0, patience=2):

        self.hidden_layer_sizes = hidden_layer_sizes
        self._n_layers = len(self.hidden_layer_sizes)
        self.activation = _preprocess_vars(activation, self._n_layers)
        self.solver = solver
        self.l2_kernel = _preprocess_vars(l2_kernel, self._n_layers + 1)
        self.l2_activation = _preprocess_vars(l2_activation, self._n_layers + 1)
        self.random_state = random_state
        super(KerasMLP, self).__init__(name="KerasMLP", max_iter=max_iter, 
                                       batch_size=batch_size, verbose=verbose,
                                       validation_fraction=validation_fraction,
                                       shuffle=shuffle, 
                                       early_stopping=early_stopping,
                                       tol=tol, patience=patience)


    def _build_model(self):
        """
        TODO
        """
        final_activation = "softmax"
        loss = "categorical_crossentropy"
        if self._n_classes == 1:
            final_activation = "sigmoid"
        if self._n_classes <= 2:
            loss = "binary_crossentropy"
        self._model = Sequential()
        for i in range(self._n_layers):
            if i == 0:
                self._model.add(Dense(self.hidden_layer_sizes[i], 
                    input_dim=self._n_attributes, 
                    activation=self.activation[i],
                    kernel_initializer=initializers.glorot_uniform(
                                                   seed=self.random_state),
                    kernel_regularizer=regularizers.l2(self.l2_kernel[i]),
                    activity_regularizer=regularizers.l2(self.l2_activation[i]))
                    )
            else:
                self._model.add(Dense(self.hidden_layer_sizes[i], 
                    activation=self.activation[i],
                    kernel_initializer=initializers.glorot_uniform(
                                                    seed=self.random_state),
                    kernel_regularizer=regularizers.l2(self.l2_kernel[i]),
                    activity_regularizer=regularizers.l2(self.l2_activation[i]))
                    )
        self._model.add(Dense(self._n_classes, activation=final_activation, 
            kernel_initializer=initializers.glorot_uniform(
                                            seed=self.random_state),
            kernel_regularizer=regularizers.l2(self.l2_kernel[self._n_layers]),
            activity_regularizer=regularizers.l2(
                                            self.l2_activation[self._n_layers]))
            )
        self._model.compile(optimizer=self.solver, 
                            loss=loss, metrics=["accuracy"])




class KerasConvolution1D(KerasBase):
    """
    TODO
    """

    def __init__(self, conv_layer_filters=(64, ), conv_kernel_size=3, 
                 conv_activation="relu", hidden_layer_sizes=(100, ), 
                 hidden_activation="relu", locally_connected=True, 
                 solver="adam", l2_kernel=0.0, l2_activation=0.0, 
                 batch_size=200, max_iter=50, shuffle=True, random_state=None, 
                 tol=0.0001, verbose=2, early_stopping=True, 
                 validation_fraction=0.0, patience=2):

        self.locally_connected = locally_connected
        self.conv_layer_filters = conv_layer_filters
        self._n_filters = len(self.conv_layer_filters)
        self.conv_kernel_size = _preprocess_vars(conv_kernel_size,
                                                 self._n_filters)
        self.hidden_layer_sizes = hidden_layer_sizes
        self._n_layers = len(self.hidden_layer_sizes)
        self.conv_activation = _preprocess_vars(conv_activation, 
                                                self._n_filters)
        self.hidden_activation = _preprocess_vars(hidden_activation,
                                                  self._n_layers)
        self.solver = solver
        self.l2_kernel = _preprocess_vars(l2_kernel, 
                                        self._n_filters + self._n_layers + 1)
        self.l2_activation = _preprocess_vars(l2_activation, 
                                        self._n_filters + self._n_layers + 1)
        self.random_state = random_state
        super(KerasConvolution1D, self).__init__(name="KerasConvolution1D", 
                                        max_iter=max_iter, 
                                        batch_size=batch_size, verbose=verbose,
                                        validation_fraction=validation_fraction,
                                        shuffle=shuffle, 
                                        early_stopping=early_stopping,
                                        tol=tol, patience=patience,
                                        is_convolution=True)

    def _build_model(self):
        """
        TODO
        """
        final_activation = "softmax"
        loss = "categorical_crossentropy"
        if self._n_classes == 1:
            final_activation = "sigmoid"
        if self._n_classes <= 2:
            loss = "binary_crossentropy"
        self._model = Sequential()
        #Convolutional Layers
        if self.locally_connected:
            for i in range(self._n_filters):
                if i == 0:
                    self._model.add(LocallyConnected1D(
                        self.conv_layer_filters[i], 
                        self.conv_kernel_sizes[i], 
                        activation=self.conv_activations[i], 
                        input_shape=(self._n_attributes,1),
                        kernel_initializer=initializers.glorot_uniform(
                                                        seed=self.random_state),
                        kernel_regularizer=regularizers.l2(self.l2_kernel[i]),
                        activity_regularizer=regularizers.l2(
                                             self.l2_activation[i]))
                    )
                else:
                    self._model.add(LocallyConnected1D(
                        self.conv_layer_filters[i], 
                        self.conv_kernel_sizes[i], 
                        activation=self.conv_activations[i], 
                        kernel_initializer=initializers.glorot_uniform(
                                                        seed=self.random_state),
                        kernel_regularizer=regularizers.l2(self.l2_kernel[i]),
                        activity_regularizer=regularizers.l2(
                                             self.l2_activation[i]))
                    )
        else:
            for i in range(self._n_filters):
                if i == 0:
                    self._model.add(Conv1D(
                        self.conv_layer_filters[i], 
                        self.conv_kernel_sizes[i], 
                        activation=self.conv_activations[i], 
                        input_shape=(self._n_attributes,1),
                        kernel_initializer=initializers.glorot_uniform(
                                                        seed=self.random_state),
                        kernel_regularizer=regularizers.l2(self.l2_kernel[i]),
                        activity_regularizer=regularizers.l2(
                                             self.l2_activation[i]))
                    )
                else:
                    self._model.add(Conv1D(
                        self.conv_layer_filters[i], 
                        self.conv_kernel_sizes[i], 
                        activation=self.conv_activations[i], 
                        kernel_initializer=initializers.glorot_uniform(
                                                        seed=self.random_state),
                        kernel_regularizer=regularizers.l2(self.l2_kernel[i]),
                        activity_regularizer=regularizers.l2(
                                             self.l2_activation[i]))
                    )
        #Dense Layers
        self._model.add(Flatten())
        for i in range(self._n_layers):
            self._model.add(Dense(self.hidden_layer_sizes[i], 
                activation=self.hidden_activations[i], 
                kernel_initializer=initializers.glorot_uniform(
                                                seed=self.random_state),
                kernel_regularizer=regularizers.l2(
                                   self.l2_kernel[self._n_filters + i]),
                activity_regularizer=regularizers.l2(
                                     self.l2_activation[self._n_filters + i]))
            )
        self._model.add(Dense(self._n_classes, activation=final_activation, 
            kernel_initializer=initializers.glorot_uniform(
                                            seed=self.random_state),
            kernel_regularizer=regularizers.l2(
                        self.l2_kernel[self._n_filters + self._n_layers]),
            activity_regularizer=regularizers.l2(
                        self.l2_activation[self._n_filters + self._n_layers]))
        )
        self._model.compile(optimizer=self.solver, 
                            loss=loss, metrics=["accuracy"])















