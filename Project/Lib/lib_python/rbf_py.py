# Import necessary packages
from ctypes import *
import numpy as np
# import matplotlib.pyplot as plt
import os

# Import the compiled Rust library
my_dll_path = os.path.dirname(os.getcwd()) + "/lib_rust/target/debug/libml_rust.so"
rust_lib = CDLL(my_dll_path)


#######################################################################################################################
def dispose_ptr(ptr):
    dispose = rust_lib.dispose_ptr
    dispose.argtype = [POINTER(c_double)]
    dispose.restype = c_void_p
    dispose(ptr)


class RustRBFModel(Structure):
    _fields_ = [("n_centroids", c_uint64),
                ("n_samples", c_uint64),
                ("n_features", c_uint64),
                ("n_outputs", c_uint64),
                ("kmeans_mode", c_bool),
                ("gamme", c_double),
                ("classification_mode", c_bool)]


class PyRBF:
    def __init__(self,
                 n_centroids: int,
                 n_samples: int,
                 n_features: int,
                 n_outputs: int,
                 classification_mode: bool,
                 kmeans_mode=True,
                 gamma=0.1):
        """Initiate an object for PyRBF
            Objects of this class contain the number of features used to predefine
            the dimension for the weight of the model.
            Each object contains also a Rust object of Rust Linear Regression.
            This Rust object is used to call for other methods from Rust lib"""
        init_rbf = rust_lib.init_rbf
        init_rbf.argtypes = [c_uint64, c_uint64,
                             c_uint64, c_uint64,
                             c_bool, c_double, c_bool]
        init_rbf.restype = POINTER(RustRBFModel)

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_centroids = n_centroids
        self.gamma = gamma
        self.kmeans_mode = kmeans_mode
        self.classification_mode = classification_mode
        self.rust_model_ = init_rbf(
            c_uint64(n_centroids),
            c_uint64(n_samples),
            c_uint64(n_features),
            c_uint64(n_outputs),
            c_bool(kmeans_mode),
            c_double(gamma),
            c_bool(classification_mode))
        # self.coef_ = self.get_weight()
        # self.bias_ = self.get_bias()

    """PyRBF is the Python wrapper for the RBF model
    defined in Rust. Users can use this Python class with normal Python types
    without converting variables to ctypes"""

    def fit(self, X, Y, norm_gamma=False):
        """Fit the model with a set of examples of inputs and outputs"""
        train_rbf = rust_lib.train_rbf
        train_rbf.argtypes = [POINTER(RustRBFModel),
                              POINTER(c_double),
                              POINTER(c_double),
                              c_uint64,
                              c_bool]
        train_rbf.restype = None

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        train_rbf(self.rust_model_,
                  np.ctypeslib.as_ctypes(X.flatten()),
                  np.ctypeslib.as_ctypes(Y.flatten()),
                  c_uint64(X.shape[0]),
                  c_bool(norm_gamma))

    def predict(self, x_test):
        """Predict new samples using the calculated weights saved in the model"""
        predict_rbf = rust_lib.predict_rbf
        predict_rbf.argtypes = [POINTER(RustRBFModel),
                                POINTER(c_double),
                                c_uint64]
        predict_rbf.restype = POINTER(c_double)

        x_test = x_test.astype('float64', copy=False)
        y_predicted = predict_rbf(self.rust_model_,
                                  np.ctypeslib.as_ctypes(x_test.flatten()),
                                  c_uint64(x_test.shape[0]))
        y_pred_np = np.ctypeslib.as_array(y_predicted, shape=(x_test.shape[0], self.n_outputs)).astype(dtype='float64')
        dispose_ptr(y_predicted)
        return y_pred_np

    def get_weight(self):
        """Get the weights calculated in the model.
        The weights is a 2D array of dim (n_centroids, n_outputs)"""
        get_weights = rust_lib.get_weights_rbf
        get_weights.argtypes = [POINTER(RustRBFModel)]
        get_weights.restype = POINTER(c_double)
        _weights = get_weights(self.rust_model_)

        weights = np.ctypeslib.as_array(_weights, shape=(self.n_centroids, self.n_outputs)).astype('float64')
        dispose_ptr(_weights)
        return weights

    def get_centroid(self):
        """Get the centroids of the model.
        The centroids is a 2D array of dim (n_centroids, n_features)"""
        get_centroids = rust_lib.get_centroids_rbf
        get_centroids.argtypes = [POINTER(RustRBFModel)]
        get_centroids.restype = POINTER(c_double)
        _centroids = get_centroids(self.rust_model_)

        centroids = np.ctypeslib.as_array(_centroids, shape=(self.n_centroids, self.n_features)).astype('float64')
        dispose_ptr(_centroids)
        return centroids

    def delete(self):
        """Delete the Python model object and also free the rust model object
            that was allocated at the heap"""
        del_RBF = rust_lib.del_RBF
        del_RBF.argtypes = [POINTER(RustRBFModel)]
        del_RBF.restype = None

        del_RBF(self.rust_model_)
        del self

    def set_mode(self, mode: bool):
        set_mode_rbf = rust_lib.set_mode_rbf
        set_mode_rbf.argtypes = [POINTER(RustRBFModel),
                                 c_bool]
        set_mode_rbf.restype = None

        set_mode_rbf(self.rust_model_, c_bool(mode))
        self.classification_mode = mode

    def set_kmeans_mode(self, kmeans_mode: bool):
        set_kmeans_mode_rbf = rust_lib.set_kmeans_mode
        set_kmeans_mode_rbf.argtypes = [POINTER(RustRBFModel),
                                 c_bool]
        set_kmeans_mode_rbf.restype = None

        set_kmeans_mode_rbf(self.rust_model_, c_bool(kmeans_mode))
        self.kmeans_mode = kmeans_mode

    def set_gamma(self, gamma: float):
        _set_gamma = rust_lib.set_gamma
        _set_gamma.argtypes = [POINTER(RustRBFModel),
                                      c_double]
        _set_gamma.restype = None

        _set_gamma(self.rust_model_, c_double(gamma))
        self.gamma = gamma


if __name__ == "__main__":
    print("RUNNING TEST CASES FOR RBF IMPLEMENTATION")
    print("=" * 50)

    print("CLASSIFICATION:")
    print("=" * 50)

    print("Linear Simple")
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

    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    lr = 0.1
    layers = [X.shape[1], 2, 1]
    mode = True
    RBF = PyRBF(layers, lr, mode)
    RBF.fit(X, Y, 100000)
    y_pred = RBF.predict(X)
    print("y_pred=", y_pred)
    print("weights=", RBF.get_weight())
