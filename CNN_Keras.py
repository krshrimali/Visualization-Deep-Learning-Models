try:
    from keras.models   import Sequential
    from keras.layers   import Dense, Dropout, Activation, Flatten
    from keras.layers   import Convolution2D, MaxPooling2D
    from keras.utils    import np_utils
    from keras.datasets import mnist
    from keras import backend as K
except ImportError:
    print("Make sure keras is installed.")

# hard coded parameters
batch_size = 128
num_classes = 10 # 0 to 9
epochs = 12


img_rows, img_cols = 28, 28

# load MNIST data into training and test dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# model architecture
model = Sequential()

model.add(Convolution2D(28, 1, 1, activation='relu', 
    input_shape = (1, 28, 28)))

# hidden layer, no need to give input shape
# keras automatically calculates shape
model.add(Convolution2D(28, (1, 1), activation='relu'))

# max pooling, matrix size (2, 2)
model.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))

# ?
model.add(Dropout(0.25))

# ?
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compilation of model
model.compile(loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 32, 
        nb_epoch = 10, verbose = 1) # verbose = ?

# evaluation of model
score = model.evaluate(x_test, y_test, verbose = 0)

# accuracy score print
print("Score: ", score)
