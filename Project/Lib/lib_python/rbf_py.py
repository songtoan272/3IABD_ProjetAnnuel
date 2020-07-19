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


class RustRBFModel(Structure):
    _fields_ = [("weights", POINTER(POINTER(c_double))),
                ("n_centroids", c_uint64),
                ("n_samples", c_uint64),
                ("n_features", c_uint64),
                ("n_outputs", c_uint64),
                ("kmeans_mode", c_bool),
                ("gamma", c_double),
                ("classification_mode", c_bool)]


class PyRBF:
    def __init__(self,
                 n_centroids: int,
                 n_samples: int,
                 n_features: int,
                 n_outputs: int,
                 classification_mode: bool,
                 gamma=1):
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

        kmeans_mode = (n_centroids != 0)

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
        _y_pred_np = np.ctypeslib.as_array(y_predicted, shape=(x_test.shape[0], self.n_outputs)).astype(dtype='float64')
        dispose_ptr(y_predicted)
        return _y_pred_np

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

    def get_centroids(self):
        """Get the centroids of the model.
        The centroids is a 2D array of dim (n_centroids, n_features)"""
        _get_centroids = rust_lib.get_centroids_rbf
        _get_centroids.argtypes = [POINTER(RustRBFModel)]
        _get_centroids.restype = POINTER(c_double)
        _centroids = _get_centroids(self.rust_model_)

        centroids = np.ctypeslib.as_array(_centroids, shape=(self.n_centroids, self.n_features)).astype('float64')
        dispose_ptr(_centroids)
        return centroids

    def delete(self):
        """Delete the Python model object and also free the rust model object
            that was allocated at the heap"""
        del_RBF = rust_lib.del_rbf
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

    def get_gamma(self):
        return self.gamma


if __name__ == "__main__":
    print("RUNNING TEST CASES FOR RBF IMPLEMENTATION")
    print("=" * 50)

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [-1, 1, -1] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [-1, -1, 1] for p in X], dtype='float64')

    n_centroids = 64
    rbf = PyRBF(n_centroids,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                n_outputs=Y.shape[1],
                gamma=50,
                classification_mode=True)
    rbf.fit(X, Y)
    y_pred = rbf.predict(X)
    nb_errors_rbf = ((y_pred - Y) != 0).sum(axis=0)

    test_points = np.array([[i, j] for i in range(100) for j in range(100)], dtype='float64') / 100 * 2 - 1

    predicted_values = rbf.predict(test_points)
    print(predicted_values)

    # plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: predicted_values[c[0]][0] > 0., enumerate(test_points)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: predicted_values[c[0]][0] > 0., enumerate(test_points)))))[:,1], color='blue', alpha=0.3, s=2)
    # plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: predicted_values[c[0]][1] > 0., enumerate(test_points)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: predicted_values[c[0]][1] > 0., enumerate(test_points)))))[:,1], color='red', alpha=0.3, s=2)
    # plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: predicted_values[c[0]][2] > 0., enumerate(test_points)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: predicted_values[c[0]][2] > 0., enumerate(test_points)))))[:,1], color='green', alpha=0.3, s=2)

    blue_points = test_points[[i for i in range(10000) if predicted_values[i, 0] > 0.], :]
    red_points = test_points[[i for i in range(10000) if predicted_values[i, 1] > 0.], :]
    green_points = test_points[[i for i in range(10000) if predicted_values[i, 2] > 0.], :]

    if len(red_points) > 0:
        plt.scatter(red_points[:, 0], red_points[:, 1], color='red', alpha=0.5, s=2)
    if len(blue_points) > 0:
        plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', alpha=0.5, s=2)
    if len(green_points) > 0:
        plt.scatter(green_points[:, 0], green_points[:, 1], color='green', alpha=0.5, s=2)

    for i in range(len(X)):
        if Y[i][0] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='blue', s=10)
        elif Y[i][1] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='red', s=10)
        elif Y[i][2] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='green', s=10)

    clusters = rbf.get_centroids()
    plt.scatter(clusters[:, 0], clusters[:, 1], color='yellow', s=25)
    plt.title("gamma=" + str(rbf.get_gamma()) + str(nb_errors_rbf))
    plt.show()
    plt.clf()

    print("nb_errors=", nb_errors_rbf)
    print("y_pred=", y_pred)
    # print("weights=", rbf.get_weight())
