# import keras module
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
# import mnist dataset
from keras.datasets import mnist
# linear stack of layers
network = Sequential()

np.random.seed(35)

num_classes = 10
network.add(Dense(512, activation='relu', input_dim=(784)))
network.add(Dropout(0.2))
network.add(Dense(512, activation='relu'))
network.add(Dropout(0.2))
network.add(Dense(num_classes, activation = 'softmax'))

network.summary()


# configuring learning process
network.compile(loss = 'categorical_crossentropy',
        optimizer = RMSprop(), metrics = ['accuracy'])
# only concerned about accuracy

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
history = network.fit(X_train, Y_train, epochs = 5, batch_size = 32, verbose = 1,
        validation_data = (X_test, Y_test))

score = network.evaluate(X_test, Y_test, verbose = 0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

