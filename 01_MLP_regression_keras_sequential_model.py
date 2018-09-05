# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 19:18:21 2018

@author: 2017B221
"""

#####################
# import data
#####################
import numpy as np

data=np.loadtxt('./housing.csv')

row, col =data.shape
np.random.shuffle(data) # This function only shuffles the array along the first axis of a multi-dimensional array. 

thr=int(row*0.8)
TrainX=data[:thr,:-1]
TrainY=data[:thr,-1]
TestX=data[thr:,:-1]
TestY=data[thr:,-1]
Num_X=col-1
#####################
# bulid model
#####################

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization

model=Sequential()
model.add(Dense(10, input_shape=(Num_X,)))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=0.99))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1))


#####################
# compile and fit
#####################

### cumstom loss
#from keras import backend as K
#def custom_loss(y_true, y_pred): #mse
#    return K.mean(K.square(y_pred - y_true), axis=-1)

### assign optimizer parm
#from keras import optimizers
#sgd = optimizers.SGD(lr = 0.01)


model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(TrainX,TrainY, batch_size= 50, epochs= 100, verbose= 1)

#####################
# evaluate
#####################

model.evaluate(TestX,TestY)

#####################
# predict
#####################
model.predict(TestX)




