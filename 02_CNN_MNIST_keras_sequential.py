
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K

print('This framework is {} \nplease be careful'.format(K.image_data_format()))

#####################
# import data
#####################


num_classes = 10
input_ch=1

# the data, split between train and test sets
(TrainX, TrainY), (TestX, TestY) = mnist.load_data()


_, input_row, input_col = TrainX.shape

TrainX=TrainX.reshape(-1,input_row, input_col, input_ch)
TestX =TestX.reshape(-1,input_row, input_col, input_ch)

TrainX = TrainX.astype('float32')
TestX = TestX.astype('float32')
TrainX /= 255 # [0-1]
TestX /= 255  # [0-1]

#####################
# bulid model
#####################

print('TrainX shape: {}'.format(TrainX.shape))
print('TestX shape: {}'.format(TestX.shape))

# one-hot-encoding
TrainY = keras.utils.to_categorical(TrainY, num_classes)
TestY = keras.utils.to_categorical(TestY, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(input_row, input_col, input_ch)))
model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), strides=(2, 2),activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(10, (1, 1), activation='sigmoid'))
model.add(Flatten())
model.summary()

#####################
# compile and fit
#####################/// 
batch_size = 128
epochs = 12

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['acc'])

model.fit(TrainX, TrainY, batch_size=batch_size, epochs=epochs,
           verbose=1,validation_data=(TestX, TestY)) #

#####################
# evaluate
#####################
score = model.evaluate(TestX, TestY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#####################
# predict
#####################


