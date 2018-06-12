import numpy as np
import pandas as pds
import time

import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plot

#import matplotlib.pyplot as plot

def load_train(path):
    height = 28
    width = 28
    train_csv = pds.read_csv(path)   #read the train data csv file
    #print(train_csv.head())

    # first column is the class label, and every following column has values for each of the 784 pixels in the image
    x = train_csv.iloc[:, 1:].values #pixel values in numpy array format
    y = train_csv.iloc[:, 0].values #labels also in numpy array format
    #print(type(x_train))

    print("TRAIN: The input shape is" + str(x.shape))
    print("TRAIN: The labels shape is" + str(y.shape))

    #convert vector to binary class matrix
    y = keras.utils.to_categorical(y)

    #split into train and validation subsets for better optimization
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 11)

    #reshape input train into images of single channel
    x_train = x_train.reshape(x_train.shape[0], height, width, 1)
    x_val = x_val.reshape(x_val.shape[0], height, width, 1)
    print("TRAIN: The train set is of shape: " + str(x_train.shape))
    print("TRAIN: The validation set is of shape: " + str(x_val.shape))

    #convert values to float32 type as required
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    #now normalize the pixels to a range of unity
    x_train /= 255
    x_val /= 255

    return x_train, x_val, y_train, y_val

def load_test(path):
    height = 28
    width = 28
    test_csv = pds.read_csv(path)

    x = test_csv.iloc[:, 1:].values
    y = test_csv.iloc[:, 0].values
    print("TEST: The input shape is" + str(x.shape))
    print("TEST: The labels shape is" + str(y.shape))

    y_test = keras.utils.to_categorical(y)

    x_test = x.reshape(x.shape[0], height, width, 1)

    x_test = x_test.astype('float32')

    x_test /= 255

    return x_test, y_test, test_csv

def conv_model(no_of_classes):
    model = Sequential()

    #starting out with a basic architecture of a CNN - 2 units of convolution layer, with each followed by a pooling layer
    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu', padding = 'same')) #first convolution unit
    model.add(MaxPooling2D(pool_size=(2,2), strides = 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))  # second convolution unit
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu')) #final dense layer for classification
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation='softmax'))  #final output layer with 10 classes

    return model

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



if __name__ == '__main__':
    start = time.time()
    classes = 10
    x_train, x_val, y_train, y_val = load_train('fashionmnist/fashion-mnist_train.csv')
    conv = conv_model(classes)

    conv.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    print(conv.summary())

    epochs = 5
    batch = 256

    history = conv.fit(x_train, y_train, batch_size = batch,
          epochs = epochs,
          verbose = 1,
          validation_data = (x_val, y_val))

    plot_accuracy(history) # plot the accuracy curves
    plot_loss(history) # plot the loss curves

    x_test, y_test, test_csv = load_test('fashionmnist/fashion-mnist_test.csv')
    evaluated = conv.evaluate(x_test, y_test, verbose=0)

    print('MAIN: Loss for Test set: ', evaluated[0])
    print('MAIN: Accuracy for Test set: ', evaluated[1])

    print("Full process took: " + str(time.time()-start) + " amount of time.")

    #get the predictions for the test data
    predicted_classes = conv.predict_classes(x_test)

    #get the indices to be plotted
    y_true = test_csv.iloc[:, 0]
    correct = np.nonzero(predicted_classes==y_true)[0]
    incorrect = np.nonzero(predicted_classes!=y_true)[0]

    targets = ["Class {}".format(i) for i in range(classes)]
    print(classification_report(y_true, predicted_classes, target_names=targets))
