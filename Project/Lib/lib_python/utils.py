import numpy as np
from Lib.lib_python.mlp_py import PyMLP
from Lib.lib_python.rbf_py import PyRBF
from Lib.lib_python.linear_regression_py import  PyLinearRegression
from Lib.lib_python.linear_classification_py import PyClassification



def confusion_matrix(self, y_true, y_pred):
    res = [[0, 0, 0] for i in range(3)]
    nb_samples = y_true.shape[0]
    for i in range(nb_samples):
        idx_true = list(y_true[i]).index(max(y_true[i]))
        idx_pred = list(y_pred[i]).index(max(y_pred[i]))
        res[idx_true][idx_pred] += 1
    return res

