'''
alex-style nnn
handle mnist data
with tensorflow -v 1.3
'''

from tensorflow.examples.tutorials.mnist import  input_data
import tensorflow as tf

minist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# hyperparameters
learning_rate = 0.001
training_state = 200000
batch_size = 64
display_step = 20

# parameters for network
n_input = 784
n_classes = 10
dropout = 0.8

# placeholder of x and y
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# conv
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)

# max pooling
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# norm
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

# whole net
def alex_net(_X, _weights, _biases, _dropout):
    # reshape to matrix
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # first block
    # conv
    conv1 = conv2d('conv1', _X, _weights['wc1'],_biases['bc1'])
    # pool
    pool1 = max_pool('pool1', conv1, k=2)
    # norm
    norm1 = norm('norm1', pool1, lsize=4)
    # dropout
    output1 = tf.nn.dropout(norm1, _dropout)

    # second block
    # conv
    conv2 = conv2d('conv2', output1, _weights['wc2'], _biases['bc2'])
    # pool
    pool2 = max_pool('pool2', conv2, k=2)
    # norm
    norm2 = norm('norm2', pool2, lsize=4)
    # dropout
    output2 = tf.nn.dropout(norm2, _dropout)

    # thrid block
    # conv
    conv3 = conv2d('conv3', output2, _weights['wc3'], _biases['bc3'])
    # pool
    pool3 = max_pool('pool3', conv3, k=2)
    # norm
    norm3 = norm('norm3', pool3, lsize=4)
    # dropout
    output3 = tf.nn.dropout(norm3, _dropout)

    # full connected layer
    dense1 = tf.reshape(output3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    #
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')

    # out
    out = tf.matmul(dense2, _weights['out']) + _biases['out']

    return out

# weights and biases

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# pred
pred = alex_net(x, weights, biases, keep_prob)

# cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

# test nerual network
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1) )
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init
init = tf.global_variables_initializer()

# begin now
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step*batch_size < training_state:
        batch_xs, batch_ys = minist.train.next_batch(batch_size)
        #
        sess.run(optimizer, feed_dict={x:batch_xs, y: batch_ys, keep_prob:1.})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Finished")
    print("Test accuracy: ", sess.run(accuracy, feed_dict={x: minist.test.images[:256], y: minist.test.labels[:256], keep_prob: 1.}))
