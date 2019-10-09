import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt

batch_size = 128
nb_classes =10
nb_epoch = 8
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv  = 3

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_train /= 255
print ('X_train shape:', X_train.shape)
print (X_train.shape[0], 'train samples')
print (X_test.shape[0], 'test samples')


Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

#i = 4600
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i, :])

model = Sequential() 
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, input_shape=(img_rows, img_cols, 1), border_mode='valid'))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)

from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
print(X_train[10].shape)
activations = activation_model.predict(X_train[10].reshape(1,28,28,1))
print(activations[0].shape)
print(activations[1].shape)
print(activations[2].shape)
print(activations[3].shape)
print(activations[4].shape)
print(activations[5].shape)
print(activations[6].shape)
print(activations[7].shape)
print(activations[8].shape)
print(activations[9].shape)
print(activations[10].shape)
print(activations[11].shape)



def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(8*2.5,8*2.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
            
            
plt.imshow(X_train[10][:,:,0]);
display_activation(activations, 16, 2, 0)
display_activation(activations, 16, 2, 1)
display_activation(activations, 16, 2, 2)
display_activation(activations, 16, 2, 3)
display_activation(activations, 16, 2, 4)
display_activation(activations, 16, 2, 5)