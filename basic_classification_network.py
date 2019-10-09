from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt

batch_size = 128
nb_classes =10
nb_epoch =20
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv  = 1024
kernel_size = 3


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_train /= 255

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols, 1), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(nb_filters, kernel_size, strides=(1,1), input_shape=(img_rows, img_cols, 1), border_mode='valid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('relu'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=2, validation_data=(X_test, Y_test))

def plot_filters(layer, x, y):
  first_layer_weights = model.layers[0].get_weights()[0]
  #print(first_layer_weights)
  print(first_layer_weights.shape)
  np.moveaxis(first_layer_weights, 1, 3)
  np.swapaxes(first_layer_weights, 1, 3)
  print(first_layer_weights.shape)
  fig = plt.figure()
  for i in range(len(first_layer_weights)):
    ax = fig.add_subplot(y, x, i+1)
    ax.matshow(first_layer_weights[i][0], cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
  plt.tight_layout()
  return plt
plot_filters(model.layers[1], 5, 6)