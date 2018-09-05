#####################
# import data
#####################
import numpy as np

data=np.loadtxt('./housing.csv')

row, col =data.shape
np.random.shuffle(data) # This function only shuffles the array along the first axis of a multi-dimensional array. 

offset=int(row*0.791)
TrainX=data[:offset,:-1]
TrainY=data[:offset,-1].reshape(-1,1)
TestX=data[offset:,:-1]
TestY=data[offset:,-1].reshape(-1,1)
Num_X=col-1
#####################
# bulid model
#####################
import tensorflow as tf

epoch_size=100
batch_size=20


train_x = tf.placeholder(tf.float32,(None,Num_X) )
train_y = tf.placeholder(tf.float32,(None,1))

hl_1= tf.layers.dense(inputs=train_x, units=10)
hl_1= tf.nn.relu(hl_1)
hl_1= tf.layers.batch_normalization(hl_1)

hl_2= tf.layers.dense(hl_1, 10, tf.nn.relu)
hl_2= tf.layers.dropout(hl_2,rate=0.8)

out= tf.layers.dense(hl_2, 1)



#####################
# compile and fit
#####################

loss = tf.losses.mean_squared_error(train_y, out)   
optimizer = tf.train.AdamOptimizer()
train_optimizer = optimizer.minimize(loss)


sess= tf.Session()

sess.run(tf.global_variables_initializer()) 
for epoch in range(epoch_size):
    for index, offset in enumerate(range(0, TrainX.shape[0], batch_size)):
        batch_x, batch_y = TrainX[offset: offset + batch_size], TrainY[offset: offset + batch_size]
        sess.run([train_optimizer], feed_dict ={train_x: batch_x, train_y: batch_y})
    # train and net output
    if epoch % 1 == 0:
        loss_val= sess.run([loss], feed_dict ={train_x: TrainX, train_y: TrainY})
        print('loss : {}'.format(loss_val))


#####################
# evaluate
#####################
loss_val = sess.run([loss], feed_dict ={train_x: TestX, train_y: TestY})
print('eval_loss : {}'.format(loss_val))
    
    
    
#####################
# predict
#####################
pred = sess.run(out, feed_dict ={train_x: TestX})    
print('predict : {}'.format(pred[:10]))        
    
sess.close()



