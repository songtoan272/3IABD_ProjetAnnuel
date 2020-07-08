# Import necessary packages
from ctypes import *
import numpy as np
import os
import sys

# Import the compiled Rust library
my_dll_path = os.path.dirname(os.getcwd()) + "/lib_rust/target/debug/libml_rust.so"
rust_lib = CDLL(my_dll_path)


#######################################################################################################################
class RustLinearRegModel(Structure):
    _fields_ = [("theta", POINTER(c_double * 1))]


def dispose_ptr(ptr):
    dispose = rust_lib.dispose_ptr
    dispose.argtype = [POINTER(c_double)]
    dispose.restype = c_void_p
    dispose(ptr)


class PyLinearRegression:
    """PyLinearRegression is the Python wrapper for the linear regression model
    defined in Rust. Users can use this Python class with normal Python types
    without converting variables to ctypes"""

    def __init__(self, nb_features):
        """Initiate an object for PyLinearRegression
            Objects of this class contain the number of features used to predefine
            the dimension for the weight of the model.
            Each object contains also a Rust object of Rust Linear Regression.
            This Rust object is used to call for other methods from Rust lib"""
        init_linear_regression_model = rust_lib.init_linear_regression_model
        init_linear_regression_model.argtypes = [c_uint64]
        init_linear_regression_model.restype = POINTER(RustLinearRegModel)

        self.nb_features_ = nb_features
        self.rust_model_ = init_linear_regression_model(c_uint64(nb_features))
        self.coef_ = self.get_weight()

    def fit(self, X, Y):
        """Fit the model with a set of examples of inputs and outputs"""
        train_linear_regression_model = rust_lib.train_linear_regression_model
        train_linear_regression_model.argtypes = [POINTER(RustLinearRegModel),
                                                  POINTER(c_double),
                                                  POINTER(c_double),
                                                  c_uint64]
        train_linear_regression_model.restype = None

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        train_linear_regression_model(self.rust_model_,
                                      np.ctypeslib.as_ctypes(X.flatten()),
                                      np.ctypeslib.as_ctypes(Y.flatten()),
                                      c_uint64(X.shape[0]))

        self.coef_ = self.get_weight()

    def predict(self, x_test):
        """Predict new samples using the calculated weights saved in the model"""
        predict_linear_regression = rust_lib.predict_linear_regression_model
        predict_linear_regression.argtypes = [POINTER(RustLinearRegModel),
                                              POINTER(c_double),
                                              c_uint64]
        predict_linear_regression.restype = POINTER(c_double)

        x_test = x_test.astype('float64', copy=False)
        _y_predicted = predict_linear_regression(self.rust_model_,
                                                 np.ctypeslib.as_ctypes(x_test.flatten()),
                                                 c_uint64(x_test.shape[0]))
        y_predicted = np.ctypeslib.as_array(_y_predicted, shape=(x_test.shape[0], 1)).astype('float64')
        dispose_ptr(_y_predicted)
        return y_predicted

    def get_weight(self):
        """Get the weights calculated in the model.
            The weight is an 2D array of dimension (n_features, )"""
        get_theta = rust_lib.get_theta_linreg_model
        get_theta.argtypes = [POINTER(RustLinearRegModel)]
        get_theta.restype = POINTER(c_double)

        _weight = get_theta(self.rust_model_)
        weight = np.ctypeslib.as_array(_weight, shape=(self.nb_features_, 1)).astype('float64')
        dispose_ptr(_weight)
        return weight

    def delete(self):
        """Delete the Python model object and also free the rust model object
            that was allocated at the heap"""
        del_linear_regression_model = rust_lib.del_linear_regression_model
        del_linear_regression_model.argtypes = [POINTER(RustLinearRegModel)]
        del_linear_regression_model.restype = None

        del_linear_regression_model(self.rust_model_)
        del self


#######################################################################################################################
class RustClassificationModel(Structure):
    _fields_ = [("theta", POINTER(c_double * 1)),
                ("alpha", c_double)]


class PyClassification:
    """PyClassification is the Python wrapper for the linear classification model
    defined in Rust. Users can use this Python class with normal Python types
    without converting variables to ctypes"""

    def __init__(self, nb_features, alpha):
        """Initiate an object/model of type PyClassification
        Objects of this class contain the number of features used to predefine
        the dimension for the weight of the model.
        Each object contains also a Rust object of Rust Linear Regression.
        This Rust object is used to call for other methods from Rust lib"""
        init_classification_model = rust_lib.init_classification_model
        init_classification_model.argtypes = [c_uint64, c_double]
        init_classification_model.restype = POINTER(RustClassificationModel)

        self.alpha_ = alpha
        self.nb_features_ = nb_features
        self.rust_model_ = init_classification_model(c_uint64(nb_features), c_double(alpha))
        self.coef_ = self.get_weight()

    def fit(self, X, Y, algorithm="rosenblatt", nb_iters=100):
        """Fit the model with a set of examples of inputs and outputs.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples, 1)
            Target values. Will be cast to X's dtype if necessary
        algorithm : a string
                    The algorithm used to calculate the weights
                    Using "rosenblatt" by default.
                    Other possible algorithms : pla
        nb_iters :  an integer
                    the number of iterations to perform the update on weights
                    100 by default
        Returns
        -------
        self : returns an instance of self."""

        if algorithm == "rosenblatt":
            train_classification_model = rust_lib.train_classification_rosenblatt
            train_classification_model.argtypes = [POINTER(RustClassificationModel),
                                                   POINTER(c_double),
                                                   POINTER(c_double),
                                                   c_uint64,
                                                   c_uint64]
            train_classification_model.restype = None
        elif algorithm == "pla":
            train_classification_model = rust_lib.train_classification_pla
            train_classification_model.argtypes = [POINTER(RustClassificationModel),
                                                   POINTER(c_double),
                                                   POINTER(c_double),
                                                   c_uint64,
                                                   c_uint64]
            train_classification_model.restype = None
        else:
            sys.exit("The algorithm passed in argument is not valid.")

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        train_classification_model(self.rust_model_,
                                   np.ctypeslib.as_ctypes(X.flatten()),
                                   np.ctypeslib.as_ctypes(Y.flatten()),
                                   c_uint64(X.shape[0]),
                                   c_uint64(nb_iters))
        self.coef_ = self.get_weight()

    def predict(self, x_test):
        """Predict new samples using the calculated weights saved in the model.

        Parameters
        ----------
        x_test : {array-like, sparse matrix} of shape (n_samples, n_features)
                Test data
        Returns
        -------
        y_predicted : array, shape (n_samples, 1)
                    Returns predicted values."""
        predict_classification = rust_lib.predict_classification_model
        predict_classification.argtypes = [POINTER(RustClassificationModel),
                                           POINTER(c_double),
                                           c_uint64]
        predict_classification.restype = POINTER(c_double)

        x_test = x_test.astype('float64', copy=False)
        _y_predicted = predict_classification(self.rust_model_,
                                             np.ctypeslib.as_ctypes(x_test.flatten()),
                                             c_uint64(x_test.shape[0]))

        y_predicted = np.ctypeslib.as_array(_y_predicted, shape=(x_test.shape[0], 1)).astype('float64')
        dispose_ptr(_y_predicted)
        return y_predicted

    def get_weight(self):
        """Get the weights calculated in the model.
        Returns
        -------
        W : array, shape (n_features, 1)
                    Returns the weights of the model"""
        get_theta = rust_lib.get_theta_classification
        get_theta.argtypes = [POINTER(RustClassificationModel)]
        get_theta.restype = POINTER(c_double)

        _weight = get_theta(self.rust_model_)
        weight = np.ctypeslib.as_array(_weight, shape=(self.nb_features_, 1)).astype('float64')
        dispose_ptr(_weight)
        return weight

    def get_alpha(self):
        """Get the learning rate of the model.
        Returns
        -------
        A : float
        Returns the learning rate of the model"""
        get_alpha_rust = rust_lib.get_alpha_classification
        get_alpha_rust.argtypes = [POINTER(RustClassificationModel)]
        get_alpha_rust.restype = c_double

        return get_alpha_rust(self.rust_model_)

    def set_alpha(self, alpha: float):
        """Set the learning rate for the model.

        Parameters
        -------
        alpha : float
                The new learning rate
        Returns
        -------
        self : returns an instance of self"""
        set_alpha_rust = rust_lib.set_alpha_classification
        set_alpha_rust.argtypes = [POINTER(RustClassificationModel)]
        set_alpha_rust.restype = c_double
        set_alpha_rust(self.rust_model_)
        self.alpha_ = self.get_alpha()

    def delete(self):
        """Delete the Python model object and also free the rust model object
        that was allocated at the heap."""
        del_classification_model = rust_lib.del_classification_model
        del_classification_model.argtypes = [POINTER(RustClassificationModel)]
        del_classification_model.restype = None

        del_classification_model(self.rust_model_)
        del self


if __name__ == "__main__":
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([
        1,
        -1,
        -1
    ])
    alpha = 0.001
    classifier = PyClassification(X.shape[1], alpha)
    classifier.fit(X, Y)
    y_pred = classifier.predict(X)
    print(y_pred)
