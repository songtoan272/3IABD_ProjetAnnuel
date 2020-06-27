import ctypes
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from os import listdir
from os.path import isfile, join
from datetime import datetime
import random
import sys

if __name__ == "__main__":

    def coherence(result, size, confiance):
        r = []
        for i in range(size):
            if result[i] <= (0 - confiance):
                r.append(-1.0)
            elif result[i] >= confiance:
                r.append(1.0)
            else:
                r.append(0)
        return r

    def data_augmentation(image):
        img = image.copy()
        width, height = image.size

        vertical_shift = True
        horizontal_shift = True
        vertical_flip = True
        horizontal_flip = True
        rotate = True
        brightness = True
        zoom = True

        # Shift
        pourcentage_shift = 0.3
        updown = 0
        side = 0
        if vertical_shift:
            updown = random.randint(0, height * pourcentage_shift)
            updown -= updown / 2
        if horizontal_shift:
            side = random.randint(0, width * pourcentage_shift)
            side -= side / 2
        img = img.transform(img.size, Image.AFFINE, (1, 0, side, 0, 1, updown))

        # Flip
        if vertical_flip:
            r = random.randint(0, 1)
            if r == 0:
                img = ImageOps.flip(img)
        if horizontal_flip:
            r = random.randint(0, 1)
            if r == 0:
                img = ImageOps.mirror(img)

        # Rotate
        if rotate:
            r = random.randint(0, 359)
            img = img.rotate(r)

        # Brightness
        if brightness:
            enhancer = ImageEnhance.Brightness(img)
            r = 1 + ((random.randint(0, 100) - 50) / 100)
            img = enhancer.enhance(r)

        # Zoom
        pourcentage_zoom = 0.3
        if zoom:
            w_r = random.randint(0, width * pourcentage_zoom) / 2
            h_r = random.randint(0, height * pourcentage_zoom) / 2
            img = img.crop((w_r, h_r, width - w_r, height - h_r))
            img = img.resize((width, height))

        return img

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

    # ------------------------------------------------------------------------
    # Définitions méthodes MLP
    # Create
    my_lib.rbf_create_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
    my_lib.rbf_create_model.restype = ctypes.c_void_p

    # Train RBF
    my_lib.rbf_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
    my_lib.rbf_train.restype = None

    # Predict classi
    my_lib.rbf_predict_classification.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    my_lib.rbf_predict_classification.restype = ctypes.POINTER(ctypes.c_double)

    # Predict regression
    my_lib.rbf_predict_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    my_lib.rbf_predict_regression.restype = ctypes.POINTER(ctypes.c_double)

    # Dispose
    my_lib.rbf_dispose.argtypes = [ctypes.c_void_p]
    my_lib.rbf_dispose.restype = None

    # Get clusters
    my_lib.rbf_get_clusters.argtypes = [ctypes.c_void_p]
    my_lib.rbf_get_clusters.restype = ctypes.POINTER(ctypes.c_double)

    # Fin load functions
    # ---------------------------------------------------------------------------

    # Load images dataset train
    dataset_train_path = "D:/Utilisateurs/Bureau/projet_annuel/dataset/train"
    images_train = [f for f in listdir(dataset_train_path) if isfile(join(dataset_train_path, f))]
    dataset_train = []
    expect_outputs_train = []

    # Data augmentation
    data_augmentation_enable = True
    images_augmented = 3

    for img in images_train:
        image = Image.open(dataset_train_path + '/' + img).convert('RGB')

        r = [-1.0, -1.0, -1.0]
        if img[:1].upper() == 'R':
            r = [1.0, -1.0, -1.0]
        elif img[:1].upper() == 'C':
            r = [-1.0, 1.0, -1.0]
        elif img[:1].upper() == 'M':
            r = [-1.0, -1.0, 1.0]

        data = np.asarray(image)
        dataset_train.append(data / 255)
        expect_outputs_train.append(r)
        if data_augmentation_enable:
            for _it in range(0, images_augmented):
                data = data_augmentation(image)
                dataset_train.append(np.asarray(data) / 255)
                expect_outputs_train.append(r)


    dataset_train = np.array(dataset_train)
    expect_outputs_train = np.array(expect_outputs_train)
    dataset_train_flattened = dataset_train.flatten()
    expect_outputs_train_flattened = expect_outputs_train.flatten()

    # sys.exit()

    # Load images dataset test
    dataset_test_path = "D:/Utilisateurs/Bureau/projet_annuel/dataset/test"
    images_test = [f for f in listdir(dataset_test_path) if isfile(join(dataset_test_path, f))]
    dataset_test = []
    expect_outputs_test = []

    for img in images_test:
        image = Image.open(dataset_test_path + '/' + img).convert('RGB')

        r = [-1.0, -1.0, -1.0]
        if img[:1].upper() == 'R':
            r = [1.0, -1.0, -1.0]
        elif img[:1].upper() == 'C':
            r = [-1.0, 1.0, -1.0]
        elif img[:1].upper() == 'M':
            r = [-1.0, -1.0, 1.0]

        expect_outputs_test.append(r)
        data = np.asarray(image)
        dataset_test.append(data / 255)

    dataset_test = np.array(dataset_test)
    expect_outputs_test = np.array(expect_outputs_test)

    data_shape_train = dataset_train.shape
    data_shape_test = dataset_test.shape
    nb_perceptrons_min = 7000
    nb_perceptrons_max = 20000
    step = 100
    alpha = 0.05
    iteration = 10000
    confiance = 0.7
    nb_hidden_layers = 2

    now = datetime.now()
    path_logs = "D:/Utilisateurs/Bureau/projet_annuel/metrics/" + now.strftime("%d_%m_%Y_%H_%M_%S") + "_metrics.txt"

    file = open(path_logs, "w+")
    file.close()

    for first_hidden in range(nb_perceptrons_min, nb_perceptrons_max + 1, step):
        layers = [data_shape_train[1] * data_shape_train[2] * data_shape_train[3]]
        for h in range(nb_hidden_layers):
            layers.append(first_hidden)
        layers.append(expect_outputs_train.shape[1])
        layers = np.array(layers)

        mlp_model = my_lib.mlp_create_model(layers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), layers.shape[0])

        now = datetime.now()
        print('Start train : ' + now.strftime("%H:%M:%S.%f"))
        my_lib.mlp_train_classification(
            mlp_model,
            dataset_train_flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            data_shape_train[0],
            data_shape_train[1] * data_shape_train[2] * data_shape_train[3],
            expect_outputs_train_flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            expect_outputs_train.shape[1],
            alpha,
            iteration
        )

        nb_error_train = 0
        for i, k in enumerate(dataset_train):
            result = my_lib.mlp_predict_classification(mlp_model, k.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            result = coherence(result, layers[len(layers) - 1], confiance)
            check = 0
            for a in range(layers[len(layers) - 1]):
                if result[a] != expect_outputs_train[i][a]:
                    check += 1
            if check != 0:
                nb_error_train += 1

        nb_error_test = 0
        for i, k in enumerate(dataset_test):
            result = my_lib.mlp_predict_classification(mlp_model, k.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            result = coherence(result, layers[len(layers) - 1], confiance)
            check = 0
            for a in range(layers[len(layers) - 1]):
                if result[a] != expect_outputs_test[i][a]:
                    check += 1
            if check != 0:
                nb_error_test += 1

        b_path_save = "D:/Utilisateurs/Bureau/projet_annuel/metrics/models".encode('utf-8')
        my_lib.mlp_save_model(mlp_model, b_path_save)

        my_lib.mlp_dispose(mlp_model)

        nb_sample_train = data_shape_train[0]
        nb_sample_test = data_shape_test[0]

        file = open(path_logs, "a")
        file.write('[' + str(layers[0]) + ', ' + str(layers[1]) + ', ' + str(layers[2]) + ', ' + str(layers[3]) + ']  alpha : ' + str(alpha) + ', it : ' + str(iteration) + '\n')
        file.write('  train_errors : ' + str(nb_error_train) + '  |  ' + str(nb_error_train / nb_sample_train * 100) + '%  |  ' + str(data_shape_train[0] - nb_error_train) + '\n')
        file.write('  test_errors : ' + str(nb_error_test) + '  |  ' + str(nb_error_test / nb_sample_test * 100) + '%  |  ' + str(data_shape_test[0] - nb_error_test) + '\n')
        file.write('-------------------------------------------------\n')
        file.close()

        now = datetime.now()
        print('Finish train : ' + now.strftime("%H:%M:%S.%f"))
        print('  [', end='')
        for l in range(layers.shape[0]):
            print(str(layers[l]) + ', ', end='')
        print(']     =>     format : (train, test)')
        print('    Nb success : (' + str(data_shape_train[0] - nb_error_train) + ', ' + str(data_shape_test[0] - nb_error_test) + ')')
        print('    Nb errors : (' + str(nb_error_train) + ', ' + str(nb_error_test) + ')')
        print("    Pourcentage de réussite : (" + str((data_shape_train[0] - nb_error_train) / nb_sample_train * 100) + '%, ' + str((data_shape_test[0] - nb_error_test) / nb_sample_test * 100) + '%)\n'
        )