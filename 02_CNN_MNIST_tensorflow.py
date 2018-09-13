
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

np.random.seed(1)

batch_size = 64
learning_rate = 0.001

mnist = input_data.read_data_sets('./mnist', one_hot=True)

data_x = tf.placeholder(tf.float32, [None,28*28])/255.                       
data_y = tf.placeholder(tf.int32, [None,10])               

image = tf.reshape(data_x,[-1,28,28,1])
conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

conv2 = tf.layers.conv2d(pool1, 32,3,1,'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

flatten = tf.reshape(pool2, [-1, 7*7*32])

output  = tf.layers.dense(flatten, 10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=data_y, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(data_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

with tf.Session() as sess:
    sess.run (tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    
    for step in range(600):
        b_x, b_y = mnist.train.next_batch(batch_size)
        _, loss_ = sess.run([train_op, loss], {data_x: b_x, data_y: b_y})
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy, {data_x: mnist.test.images, data_y: mnist.test.labels})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
            
            
test_output = sess.run(output, {data_x: mnist.test.images})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(mnist.test.labels, 1), 'real number')