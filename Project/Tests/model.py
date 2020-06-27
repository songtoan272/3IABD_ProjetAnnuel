#!/usr/bin/env python

import numpy as np
import ctypes
import sys
from PIL import Image

if __name__ == "__main__":

    # Chargement lib
    path_to_dll = "D:/lib_rust/libC/cmake-build-debug/libC.dll"

    my_lib = ctypes.CDLL(path_to_dll)

    # ------------------------------------------------------------------------
    # Définitions méthodes modèle linéaire
    # Create
    my_lib.linear_create_model.argtypes = [ctypes.c_int]
    my_lib.linear_create_model.restype = ctypes.c_void_p

    # Predict regression
    my_lib.linear_predict_model_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_classification.restype = ctypes.c_double

    # Train regression
    my_lib.linear_train_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int
    ]
    my_lib.linear_train_model_classification.restype = None

    # Predict classification
    my_lib.linear_predict_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_classification.restype = ctypes.c_double

    # Train classification
    my_lib.linear_train_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.linear_train_model_classification.restype = None

    # Dispose
    my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p, ctypes.c_int]
    my_lib.linear_dispose_model.restype = None

    # ------------------------------------------------------------------------
    # Définitions méthodes MLP
    # Create
    my_lib.mlp_create_model.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    my_lib.mlp_create_model.restype = ctypes.c_void_p

    # Predict regression
    my_lib.mlp_predict_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double)
    ]
    my_lib.mlp_predict_regression.restype = ctypes.POINTER(ctypes.c_double)

    # Train regression
    my_lib.mlp_train_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.mlp_train_regression.restype = None

    # Predict classification
    my_lib.mlp_predict_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double)
    ]
    my_lib.mlp_predict_classification.restype = ctypes.POINTER(ctypes.c_double)

    # Train classification
    my_lib.mlp_train_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.mlp_train_classification.restype = None

    # Dispose
    my_lib.mlp_dispose.argtypes = [ctypes.c_void_p]
    my_lib.mlp_dispose.restype = None

    # Save model
    my_lib.mlp_save_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    my_lib.mlp_save_model.restype = None

    # Load model
    my_lib.mlp_load_model.argtypes = [ctypes.c_char_p]
    my_lib.mlp_load_model.restype = ctypes.c_void_p

    type_model = sys.argv[1]
    model_path = sys.argv[2].encode('utf-8')
    img_path = sys.argv[3]

    image = Image.open(img_path).convert('RGB')
    data = np.asarray(image).flatten() / 255

    if type_model == 'mlp':
        model = my_lib.mlp_load_model(model_path)

        result = my_lib.mlp_predict_classification(model, data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        my_lib.mlp_dispose(model)

        print(result)
