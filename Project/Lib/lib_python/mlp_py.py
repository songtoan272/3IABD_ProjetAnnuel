# Import necessary packages
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# from Lib.lib_python.utils import *

# Import the compiled Rust library
my_dll_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lib_rust/target/release/libml_rust.so"
# print(my_dll_path)
rust_lib = CDLL(my_dll_path)


#######################################################################################################################
def dispose_ptr(ptr):
    dispose = rust_lib.dispose_ptr
    dispose.argtype = [POINTER(c_double)]
    dispose.restype = c_void_p
    dispose(ptr)


class RustMLPModel(Structure):
    _fields_ = [("weights", POINTER(POINTER(POINTER(c_double)))),
                ("bias", POINTER(POINTER(POINTER(c_double)))),
                ("n_layers", c_uint64),
                ("layer_sizes", POINTER(c_uint64)),
                ("learning_rate", c_double),
                ("n_iters", c_uint64),
                ("classification_mode", c_bool)]


class PyMLP:
    """PyMLP is the Python wrapper for the MLP model
    defined in Rust. Users can use this Python class with normal Python types
    without converting variables to ctypes"""

    def __init__(self,
                 layer_sizes: list,
                 learning_rate: float,
                 classification_mode: bool):
        """Initiate an object for PyMLP
            Objects of this class contain the number of features used to predefine
            the dimension for the weight of the model.
            Each object contains also a Rust object of Rust Linear Regression.
            This Rust object is used to call for other methods from Rust lib"""
        init_mlp = rust_lib.init_mlp
        init_mlp.argtypes = [POINTER(c_uint64), c_uint64, c_double, c_bool]
        init_mlp.restype = POINTER(RustMLPModel)

        self.n_features = layer_sizes[0]
        self.n_outputs = layer_sizes[-1]
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.lr = learning_rate
        self.classification_mode = classification_mode
        self.rust_model_ = init_mlp(
            np.ctypeslib.as_ctypes(np.array(layer_sizes, dtype=c_uint64)),
            c_uint64(self.n_layers),
            c_double(learning_rate),
            c_bool(classification_mode)
        )
        # self.coef_ = self.get_weight()
        # self.bias_ = self.get_bias()

    def fit(self, X, Y, nb_iters: int):
        """Fit the model with a set of examples of inputs and outputs.
        The model picks randomly a sample to update its weights for each iteration."""
        train_mlp = rust_lib.train_mlp
        train_mlp.argtypes = [POINTER(RustMLPModel),
                              POINTER(c_double),
                              POINTER(c_double),
                              c_uint64,
                              c_uint64]
        train_mlp.restype = None

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        train_mlp(self.rust_model_,
                  np.ctypeslib.as_ctypes(X.flatten()),
                  np.ctypeslib.as_ctypes(Y.flatten()),
                  c_uint64(X.shape[0]),
                  c_uint64(nb_iters))

        self.coef_ = self.get_weight()
        self.bias_ = self.get_bias()

    def fit_retrieve_metrics(self, X, Y, X_val, Y_val, nb_iters: int):
        """Fit the model with a set of examples of inputs and outputs.
        The model picks randomly a sample to update its weights every iteration.
        Return loss and accuracy of train dataset and validation dataset
        for each iteration."""
        train_mlp_metrics = rust_lib.train_mlp_return_metrics
        train_mlp_metrics.argtypes = [POINTER(RustMLPModel),
                                      POINTER(c_double),
                                      POINTER(c_double),
                                      c_uint64,
                                      POINTER(c_double),
                                      POINTER(c_double),
                                      c_uint64,
                                      c_uint64]
        train_mlp_metrics.restype = POINTER(c_double)

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        X_val = X_val.astype('float64', copy=False)
        Y_val = Y_val.astype('float64', copy=False)
        _metrics = train_mlp_metrics(
            self.rust_model_,
            np.ctypeslib.as_ctypes(X.flatten()),
            np.ctypeslib.as_ctypes(Y.flatten()),
            c_uint64(X.shape[0]),
            np.ctypeslib.as_ctypes(X_val.flatten()),
            np.ctypeslib.as_ctypes(Y_val.flatten()),
            c_uint64(X_val.shape[0]),
            c_uint64(nb_iters))

        metrics = np.ctypeslib.as_array(_metrics, shape=(nb_iters, 4)).astype(dtype='float64')
        dispose_ptr(_metrics)
        return metrics

    def fit_epoch(self, X, Y, X_val, Y_val):
        """Train the model for one epoch. Each epoch, the model uses every sample of dataset once
        to update its weights.

        Return: an array of metrics contains [train_loss, train_accuracy, val_loss, val_accuracy]"""
        train_epoch = rust_lib.train_epoch_mlp
        train_epoch.argtypes = [POINTER(RustMLPModel),
                                POINTER(c_double),
                                POINTER(c_double),
                                c_uint64,
                                POINTER(c_double),
                                POINTER(c_double),
                                c_uint64]
        train_epoch.restype = POINTER(c_double)

        X = X.astype('float64', copy=False)
        Y = Y.astype('float64', copy=False)
        X_val = X_val.astype('float64', copy=False)
        Y_val = Y_val.astype('float64', copy=False)
        _metrics = train_epoch(
            self.rust_model_,
            np.ctypeslib.as_ctypes(X.flatten()),
            np.ctypeslib.as_ctypes(Y.flatten()),
            c_uint64(X.shape[0]),
            np.ctypeslib.as_ctypes(X_val.flatten()),
            np.ctypeslib.as_ctypes(Y_val.flatten()),
            c_uint64(X_val.shape[0]))

        metrics = np.ctypeslib.as_array(_metrics, shape=(4,)).astype(dtype='float64')
        dispose_ptr(_metrics)
        return metrics.tolist()

    def predict(self, x_test, convert=False):
        """Predict new samples using the calculated weights saved in the model"""
        predict_mlp = rust_lib.predict_mlp
        predict_mlp.argtypes = [POINTER(RustMLPModel),
                                POINTER(c_double),
                                c_uint64]
        predict_mlp.restype = POINTER(c_double)

        x_test = x_test.astype('float64', copy=False)
        y_predicted = predict_mlp(self.rust_model_,
                                  np.ctypeslib.as_ctypes(x_test.flatten()),
                                  c_uint64(x_test.shape[0]))
        _y_pred_np = np.ctypeslib.as_array(y_predicted, shape=(x_test.shape[0], self.n_outputs)).astype(dtype='float64')
        dispose_ptr(y_predicted)

        def convert_row(row):
            m = max(row)
            res = np.where(row == m, 1., -1.)
            # res = np.where(row > 0.0, 1., -1.)
            return res

        if convert:
            return np.array([convert_row(pred) for pred in _y_pred_np])
        else:
            return _y_pred_np

    def get_weight(self):
        """Get the weights calculated in the model.
            The weight is a List of 2D arrays"""
        get_weights = rust_lib.get_weights_mlp
        get_weights.argtypes = [POINTER(RustMLPModel)]
        get_weights.restype = POINTER(c_double)
        _weights = get_weights(self.rust_model_)

        # Transform to a List of 2D np Arrays

        # Calculate the total number of weights and
        # the nb of weights in between 2 layers
        nb_w = []
        for i in range(self.n_layers - 1):
            nb_w.append(self.layer_sizes[i] * self.layer_sizes[i + 1])
        weights = np.ctypeslib.as_array(_weights, shape=(sum(nb_w),)).astype('float64')
        dispose_ptr(_weights)

        # Reconstruct the weights from the 1D ndarray
        list_w: list[np.ndarray] = []
        _start = 0
        for i in range(self.n_layers - 1):
            list_w.append(np.reshape(weights[_start:_start + nb_w[i]],
                                     (self.layer_sizes[i], self.layer_sizes[i + 1])))
            _start += nb_w[i]

        return list_w

    def get_bias(self):
        """Get the bias calculated in the model.
            The weight is a List of 1D arrays"""
        get_bias = rust_lib.get_bias_mlp
        get_bias.argtypes = [POINTER(RustMLPModel)]
        get_bias.restype = POINTER(c_double)
        _bias = get_bias(self.rust_model_)

        # Transform to a List of 1D np Arrays
        bias = np.ctypeslib.as_array(_bias, shape=(sum(self.layer_sizes[1:]),)).astype('float64')
        dispose_ptr(_bias)

        # Reconstruct the bias from the 1D npArray
        list_b: list[np.ndarray] = []
        _start = 0
        for i in range(1, self.n_layers):
            list_b.append(np.reshape(bias[_start:_start + self.layer_sizes[i]],
                                     (self.layer_sizes[i],)))
            _start += self.layer_sizes[i]

        return list_b

    def confusion_matrix(self, y_true, y_pred):
        n = self.n_outputs
        res = [[0 for i in range(n)] for j in range(n)]
        nb_samples = y_true.shape[0]
        for i in range(nb_samples):
            idx_true = list(y_true[i]).index(max(y_true[i]))
            idx_pred = list(y_pred[i]).index(max(y_pred[i]))
            res[idx_true][idx_pred] += 1
        return res

    def delete(self):
        """Delete the Python model object and also free the rust model object
            that was allocated at the heap"""
        del_mlp = rust_lib.del_mlp
        del_mlp.argtypes = [POINTER(RustMLPModel)]
        del_mlp.restype = None

        del_mlp(self.rust_model_)
        del self

    def set_mode(self, mode: bool):
        set_mode_mlp = rust_lib.set_mode_mlp
        set_mode_mlp.argtypes = [POINTER(RustMLPModel),
                                 c_bool]
        set_mode_mlp.restype = None

        set_mode_mlp(self.rust_model_, c_bool(mode))
        self.classification_mode = mode

    def set_learning_rate(self, lr: float):
        set_learning_rate = rust_lib.set_learning_rate
        set_learning_rate.argtypes = [POINTER(RustMLPModel),
                                      c_double]
        set_learning_rate.restype = None

        set_learning_rate(self.rust_model_, c_double(lr))
        self.lr = lr

    def save_model(self, filename):
        save_path = os.getcwd() + f"/models/{filename}.txt"
        with open(save_path, "w") as f:
            f.write(str(self.n_layers))
            f.write("\n")
            f.write(str(self.layer_sizes))
            f.write("\n")
            f.write(str(self.lr))
            f.write("\n")
            for w in self.get_weight():
                for x in w.flatten():
                    f.write("%1.5f " %x)
                f.write("\n")
            for b in self.get_bias():
                f.write(str(b))
                f.write("\n")
        # filehandler = open(save_path, 'w')
        # pickle.dump(self, filehandler)
        # filehandler.close()

    @staticmethod
    def load_model(path):
        filehandler = open(path, 'r')
        mlp = pickle.load(filehandler)
        filehandler.close()
        return mlp

    def __str__(self, show_weights=False):
        res = f"number of layers: {self.n_layers}\n"
        res += f"layers: {self.layer_sizes}\n"
        res += f"learning rate: {self.lr}\n"
        if show_weights:
            res += "weights:\n"
            for w in self.coef_:
                res += str(w) + "\n"
            res += str(self.bias_) +"\n"
        return res


if __name__ == "__main__":
    print("RUNNING TEST CASES FOR MLP IMPLEMENTATION")
    print("=" * 50)

    print("CLASSIFICATION:")
    print("=" * 50)

    print("Linear Simple")
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [-1, -1, -1] for p in X], dtype='float64')

    # X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    # Y = np.array([1, 1, -1, -1])

    lr = 0.1
    layers = [X.shape[1], Y.shape[1]]
    mode = True
    mlp = PyMLP(layers, lr, mode)
    metrics = mlp.fit_epoch(X, Y, X, Y)
    print('{: <20}'.format('Train Loss') + '  |  ' + '{: <20}'.format('Train Acc') + '  |  ' +
          '{: <20}'.format('Val Loss') + '  |  ' + '{: <20}'.format('Val Acc'))
    for i in range(1):
        print('{: <20}'.format(metrics[i, 0]) + '  |  ' + '{: <20}'.format(metrics[i, 1]) + '  |  ' +
              '{: <20}'.format(metrics[i, 2]) + '  |  ' + '{: <20}'.format(metrics[i, 3]))
    y_pred = mlp.predict(X)
    y_pred_converted = mlp.predict(X, convert=True)
    # y_pred_translated = np.array(
    #     [[1, -1, -1] if pred[0] == max(pred) else
    #      [-1, 1, -1] if pred[1] == max(pred) else
    #      [-1, -1, 1] if pred[2] == max(pred) else
    #      [-1, -1, -1]
    #      for pred in y_pred], dtype='float64')
    # print("y_pred=", y_pred)
    # print("y_pred_trans=", y_pred_translated)

    for i in range(X.shape[0]):
        if not (Y[i] == y_pred_converted[i]).all():
            print('{: <20}'.format(str(Y[i])) + '  |  ' + '{: <20}'.format(str(y_pred_converted[i])) +
                  '  |  ' + '{: <20}'.format(str(y_pred[i].tolist())))
    print(((y_pred_converted - Y) != 0).sum(axis=0))
    print(mlp.confusion_matrix(Y, y_pred_converted))
    print("weights=", mlp.get_weight())
