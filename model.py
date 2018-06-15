from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class Model:
    def create_model(self, no_of_classes):
        self.model = Sequential()

        #now trying an architecture of a CNN - 3 units of 1 convolution layer each, with each followed by a pooling layer, followed by dense layer for classification
        #for normalized datasets, deep networks are not required, so multiple convolution layers can be stacked, but that was issue with model v2 commit
        self.model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same')) #first convolution unit
        self.model.add(MaxPooling2D(pool_size=(2,2), strides = 2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))  # second convolution unit
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))  # third convolution unit
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'relu')) #final dense layer for classification
        self.model.add(Dropout(0.5))
        self.model.add(Dense(no_of_classes, activation = 'softmax'))  #final output layer with 10 classes

        return self.model
