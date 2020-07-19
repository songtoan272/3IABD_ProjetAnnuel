# Import necessary packages
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the compiled Rust library
my_dll_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib_rust/target/release/libml_rust.so"
rust_lib = CDLL(my_dll_path)


#######################################################################################################################
def dispose_ptr(ptr):
    dispose = rust_lib.dispose_ptr
    dispose.argtype = [POINTER(c_double)]
    dispose.restype = c_void_p
    dispose(ptr)


class RustSVMModel(Structure):
    _fields_ = [("weights", POINTER(c_double)),
                ("bias", c_double),
                ("alpha", POINTER(c_double)),
                ("n_features", c_uint64)]


class PySVM:
    def __init__(self, n_features: int):
        """Initiate an object for PySVM
            Objects of this class contain the number of features used to predefine
            the dimension for the weight of the model.
            Each object contains also a Rust object of Rust Linear Regression.
            This Rust object is used to call for other methods from Rust lib"""
        init_svm = rust_lib.init_svm
        init_svm.argtypes = [c_uint64]
        init_svm.restype = POINTER(RustSVMModel)

        self.n_features = n_features
        self.rust_model_ = init_svm(c_uint64(n_features))
        # self.coef_ = self.get_weight()
        # self.bias_ = self.get_bias()

    """PySVM is the Python wrapper for the SVM model
    defined in Rust. Users can use this Python class with normal Python types
    without converting variables to ctypes"""

    def fit(self, X, Y, norm_gamma=False):
        """Fit the model with a set of examples of inputs and outputs"""
        train_svm = rust_lib.train_svm
        train_svm.argtypes = [POINTER(RustSVMModel),
                              POINTER(c_double),
                              POINTER(c_double),
                              c_uint64]
        train_svm.restype = None

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        train_svm(self.rust_model_,
                  np.ctypeslib.as_ctypes(X.flatten()),
                  np.ctypeslib.as_ctypes(Y.flatten()),
                  c_uint64(X.shape[0]))

    def predict(self, x_test):
        """Predict new samples using the calculated weights saved in the model"""
        predict_svm = rust_lib.predict_svm
        predict_svm.argtypes = [POINTER(RustSVMModel),
                                POINTER(c_double),
                                c_uint64]
        predict_svm.restype = POINTER(c_double)

        x_test = x_test.astype('float64', copy=False)
        y_predicted = predict_svm(self.rust_model_,
                                  np.ctypeslib.as_ctypes(x_test.flatten()),
                                  c_uint64(x_test.shape[0]))
        y_pred_np = np.ctypeslib.as_array(y_predicted, shape=(x_test.shape[0], 1)).astype(dtype='float64')
        dispose_ptr(y_predicted)
        return y_pred_np

    def get_weight(self):
        """Get the weights calculated in the model.
        The weights is a 1D array of dim (n_features,)"""
        get_weights = rust_lib.get_weights_svm
        get_weights.argtypes = [POINTER(RustSVMModel)]
        get_weights.restype = POINTER(c_double)
        _weights = get_weights(self.rust_model_)

        weights = np.ctypeslib.as_array(_weights, shape=(self.n_features, 1)).astype('float64')
        dispose_ptr(_weights)
        return weights

    def get_bias(self):
        get_bias = rust_lib.get_bias_svm
        get_bias.argtypes = [POINTER(RustSVMModel)]
        get_bias.restype = c_double
        return float(get_bias(self.rust_model_))

    def delete(self):
        """Delete the Python model object and also free the rust model object
            that was allocated at the heap"""
        del_SVM = rust_lib.del_svm
        del_SVM.argtypes = [POINTER(RustSVMModel)]
        del_SVM.restype = None

        del_SVM(self.rust_model_)
        del self


if __name__ == "__main__":
    print("RUNNING TEST CASES FOR SVM IMPLEMENTATION")
    print("=" * 50)

    X = np.array([
        [0.5, 0.7],
        [0.4, 0.5],
        [0.6, 0.6],
        [0.1, 0.7],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.8, 0.6],
        [0.9, 0.3],
        [0.8, 0.1],
        [0.3, 0.1]
    ])
    Y = np.array([
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ])

    svm = PySVM(X.shape[0])
    svm.fit(X, Y)
    y_pred = svm.predict(X)
    nb_errors_svm = ((y_pred - Y) != 0).sum()

    test_points = np.array([[i, j] for i in range(100) for j in range(100)], dtype='float64') / 100

    test_points = np.array([[i, j] for i in range(100) for j in range(100)], dtype='float64') / 100

    predicted_values = svm.predict(test_points).flatten()

    red_points = test_points[(predicted_values < 0.0)]
    blue_points = test_points[(predicted_values > 0.0)]
    print("red=", red_points.shape)
    print("blue=", blue_points.shape)

    if len(red_points) > 0:
        plt.scatter(red_points[:, 0], red_points[:, 1], color='red', alpha=0.5, s=2)
    if len(blue_points) > 0:
        plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', alpha=0.5, s=2)

    plt.scatter(X[0:5, 0], X[0:5, 1], color='blue', s=50)
    plt.scatter(X[5:10, 0], X[5:10, 1], color='red', s=50)
    plt.show()
    plt.clf()

    print("nb_errors=", nb_errors_svm)
    print("y_pred=", y_pred)
    # print("weights=", svm.get_weight())
