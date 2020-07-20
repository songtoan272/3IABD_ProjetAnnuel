from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from Lib.lib_python.mlp_py import PyMLP

# DEFINE GLOBAL PARAMETERS OF MODEL
BATCH_SIZE = 1
EPOCHS = 140
SUFFLE = True
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
COLOR_MODE = 'rgb'
CLASS_NAMES = np.array(['makise', 'cc', 'rem'])
NB_TRAIN = 5000
NB_VAL = 500

# Data dir
train_dir = f"../../Dataset/{IMG_HEIGHT}x{IMG_WIDTH}/classic/train"
test_dir = f"../../Dataset/{IMG_HEIGHT}x{IMG_WIDTH}/classic/test"

# Data Generator
train_image_generator = ImageDataGenerator(rotation_range=30,
                                           width_shift_range=0.3,
                                           height_shift_range=0.3,
                                           brightness_range=(0.5, 1.5),
                                           zoom_range=0.3,
                                           horizontal_flip=True,
                                           rescale=1. / 255)
validation_image_generator = ImageDataGenerator(horizontal_flip=True,
                                                zoom_range=0.3,
                                                rescale=1. / 255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           color_mode=COLOR_MODE,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           classes=list(CLASS_NAMES),
                                                           class_mode='categorical')
validation_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                     directory=test_dir,
                                                                     shuffle=True,
                                                                     color_mode=COLOR_MODE,
                                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                     classes=list(CLASS_NAMES),
                                                                     class_mode='categorical')


# Generate a dataset with train and validation sets
def data_generator(train_data_gen, validation_data_gen, nb_img_train, nb_img_val):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for i in range(nb_img_train):
        image_batch, label_batch = next(train_data_gen)
        x_train.append(image_batch[0].flatten())
        y_train.append(np.where(label_batch[0].flatten() == 1., 1., -1.))

        if i < nb_img_val:
            image_batch, label_batch = next(validation_data_gen)
            x_val.append(image_batch[0].flatten())
            y_val.append(np.where(label_batch[0].flatten() == 1., 1., -1.))
    x_train = np.array(x_train, copy=False)
    y_train = np.array(y_train, copy=False)
    x_val = np.array(x_val, copy=False)
    y_val = np.array(y_val, copy=False)
    return (x_train, y_train, x_val, y_val)


x_train, y_train, x_val, y_val = data_generator(train_data_gen, validation_data_gen, NB_TRAIN, NB_VAL)
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)
print("Data generation done")

# Create and train MLP model
def train_model(model, xtrain, ytrain, xval, yval, epochs, suffle=False):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    time = []
    start = datetime.now()
    for i in range(epochs):
        if (suffle):
            xtrain, ytrain = shuffle(xtrain, ytrain)
        metric = model.fit_epoch(xtrain, ytrain, xval, yval)
        end = datetime.now()
        diff = (end-start).total_seconds()
        train_loss.append(metric[0])
        train_acc.append(metric[1])
        val_loss.append(metric[2])
        val_acc.append(metric[3])
        time.append(diff)
        print(f"Epoch {i+1}: {str(metric)}")
    return train_loss, train_acc, val_loss, val_acc, time


def plot_metrics(metrics, save_path):
    train_loss, train_acc, val_loss, val_acc, time = metrics
    epoch_val = [i for i in range(1, len(train_loss) + 1)]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6.4, 12.8))
    ax1.plot(epoch_val, train_loss, c="red", label="Train")
    ax1.plot(epoch_val, val_loss, c="green", label="Validation")
    ax1.legend(loc='upper right')
    ax1.set_title("Loss evaluation")
    ax2.plot(epoch_val, train_acc, c="blue", label="Train")
    ax2.plot(epoch_val, val_acc, c="yellow", label="Validation")
    ax2.legend(loc='lower right')
    ax2.set_title("Accuracy evaluation")
    ax3.plot(epoch_val, time, c="black")
    ax3.set_title("Time execution")
    plt.show()
    fig.savefig(save_path)


def save_metrics(self, metrics, path, conf_mats=None):
    train_loss, train_acc, val_loss, val_acc, time_exe = metrics
    with open(path, "w") as f:
        f.write("MLP\n")
        f.write(str(mlp.layer_sizes))
        f.write("\n")
        f.write(f"learning rate: {mlp.lr}\n")
        f.write(f"Epochs: {len(train_loss)}\n")
        f.write(f"Shuffle: {SUFFLE}\n")
        f.write(f"Image sizes: {IMG_WIDTH} x {IMG_HEIGHT}\n")
        f.write(f"Time execution: {time_exe[-1]} seconds\n\n")
        for u in range(len(train_loss)):
            line = str(u + 1) + '  -  Loss: ' + '{:.5f}'.format(train_loss[u]) + \
                   '     - Accurate: ' + '{:.5f}'.format(train_acc[u]) + '    |'
            line += '    -  Loss: ' + '{:.5f}'.format(val_loss[u]) + \
                    '    - Accurate: ' + '{:.5f}'.format(val_acc[u]) + "\n"
            f.write(line)
        train_conf_mat, val_conf_mat = conf_mats
        f.write('Train ----------------------------------------\n')
        for i in range(mlp.n_outputs):
            line = ""
            for j in range(mlp.n_outputs):
                line += '{:>10}'.format(train_conf_mat[i][j])
                line += "    "
            line += "\n"
            f.write(line)
        f.write('----------------------------------------------\n')
        f.write('Test -----------------------------------------\n')
        for i in range(mlp.n_outputs):
            line = ""
            for j in range(mlp.n_outputs):
                line += '{:>10}'.format(val_conf_mat[i][j])
                line += "    "
            line += "\n"
            f.write(line)
        f.write('----------------------------------------------')


if __name__ == "__main__":
    # Create
    layer_sizes = [x_train.shape[1], 300, 300, y_train.shape[1]]
    lr = 0.0008
    mlp = PyMLP(layer_sizes, lr, True)

    #Train
    metrics = train_model(mlp, x_train, y_train, x_val, y_val, EPOCHS, suffle=SUFFLE)
    print("Train done")

    #Predict
    y_pred_train = mlp.predict(x_train, convert=True)
    y_pred_val = mlp.predict(x_val, convert=True)
    print(y_pred_train.shape)
    print(y_pred_val.shape)
    print("Predict done")

    #Confustion matrix
    confu_train = mlp.confusion_matrix(y_train, y_pred_train)
    confu_val = mlp.confusion_matrix(y_val, y_pred_val)

    #Plot and save metrics
    filename = f"self-{IMG_WIDTH}x{IMG_HEIGHT}-0008-shuffled-fixed"
    save_path = os.getcwd() + f"/metrics/{filename}"
    plot_metrics(metrics, save_path+".png")
    save_metrics(mlp, metrics, save_path+".txt", conf_mats=(confu_train, confu_val))
    print("Plot and save done")

    #Save and load model
    save_path = os.getcwd() + f"/models/{filename}.mlp"
    mlp.save_model(filename)
    # loaded_mlp = mlp.load_model(save_path)
    # print(loaded_mlp)