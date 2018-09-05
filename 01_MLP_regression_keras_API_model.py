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

from keras.models import Model
from keras.layers import Input, Activation, Dense, Dropout, BatchNormalization

Inputs = Input(shape=(Num_X,))
x = Dense(10)(Inputs)
x = Activation('relu')(x)
x = BatchNormalization(momentum=0.99)(x)
x = Dense(10, activation='relu')(x)
x = Dropout(0.8)(x)
Outputs = Dense(1)(x)

model=Model(inputs=Inputs, outputs=Outputs)

#####################
# compile and fit
#####################

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(TrainX,TrainY, batch_size= 50, epochs= 100, verbose= 1)

#####################
# evaluate
#####################

model.evaluate(TestX,TestY)

#####################
# predict
#####################
model.predict(TestX)




