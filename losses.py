import pandas as pds
import numpy as np
from keras.utils import to_categorical

from sklearn.metrics import classification_report

import matplotlib.pyplot as plot


def load_test(path, height, width):
    # load test data
    test_csv = pds.read_csv(path)

    x = test_csv.iloc[:, 1:].values
    y = test_csv.iloc[:, 0].values
    print("TEST: The input shape is" + str(x.shape))
    print("TEST: The labels shape is" + str(y.shape))

    y_test = to_categorical(y)

    x_test = x.reshape(x.shape[0], height, width, 1)

    x_test = x_test.astype('float32')

    x_test /= 255.0

    print("TEST: Test data has been processed and loaded")

    return x_test, y_test, test_csv

def plot_accuracy(fit):
    plot.figure(figsize = [8, 8])
    plot.plot(fit.history['val_acc'],'g',linewidth = 2.0)
    plot.plot(fit.history['acc'],'r', linewidth = 2.0)
    plot.legend(['Training Accuracy', 'Validation Accuracy'], fontsize = 14)
    plot.xlabel('epochs ', fontsize = 14)
    plot.ylabel('accuracy', fontsize = 14)
    plot.title('accuracy vs epochs', fontsize = 14)
    plot.show()


def plot_loss(fit):
    plot.figure(figsize = [8, 8])
    plot.plot(fit.history['val_loss'],'g', linewidth = 2.0)
    plot.plot(fit.history['loss'],'r', linewidth = 2.0)
    plot.legend(['Training loss', 'Validation Loss'], fontsize = 14)
    plot.xlabel('epochs ', fontsize = 14)
    plot.ylabel('loss', fontsize = 14)
    plot.title('loss vs epochs', fontsize = 14)
    plot.show()

def prediction(conv, x_test, test, classes):
    predicted = conv.predict_classes(x_test)

    #get all samples for plotting
    y_actual = test.iloc[:, 0]
    correct = np.nonzero(predicted == y_actual)[0]
    incorrect = np.nonzero(predicted != y_actual)[0]

    targets = ["Class {}".format(i) for i in range(classes)]

    print("PREDICTION IS:")
    print(classification_report(y_actual, predicted, target_names = targets))
