try:
    import keras
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

if K.image_data_format() == 'channels_first':
    print("original shapes: " + str(x_train.shape) + ", " + str(x_test.shape))
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0]  , 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols) # channels first 
    print("new shapes: " + str(x_train.shape) + ", " + str(x_test.shape))
else:
    print("original shapes: " + str(x_train.shape) + ", " + str(x_test.shape))
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test  = x_test.reshape(x_test.shape[0]  , img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1) # channels first 
    print("new shapes: " + str(x_train.shape) + ", " + str(x_test.shape))

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# # get data from 0 to 1 (pixels)
# x_train /= 255
# x_test /= 255

print('x_train shape: ', x_train.shape)
print('Number of training samples: ', x_train.shape[0])
print('Number of test samples: ', x_train.shape[0])

# class vectors to binary class vectors
print(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0])
# model architecture
model = Sequential()

model.add(Convolution2D(28, kernel_size = (1, 1), activation='relu', 
    input_shape = input_shape))

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
model.add(Dense(num_classes, activation='softmax'))

# compilation of model
model.compile(loss = keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adadelta(),
        metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, 
        nb_epoch = epochs, verbose = 1, validation_data = (x_test, y_test)) # verbose = ?

# evaluation of model
score = model.evaluate(x_test, y_test, verbose = 0)

# accuracy score print
print("Score: ", score)
