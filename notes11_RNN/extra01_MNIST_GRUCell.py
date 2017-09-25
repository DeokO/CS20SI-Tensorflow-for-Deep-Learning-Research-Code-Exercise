# 이 코드는 RNN에 대해 익히기 위해 golbin님의 git에있는 코드를 참조해서 응용
# GRU cell을 이용하고, RNN을 2층으로 쌓고 Dropout을 적용
# Fully connected는 tf.contrib.layers를 이용, dropout
# https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/01%20-%20MNIST.py

##############################################################################
# import module
##############################################################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)



##############################################################################
# Hyper-parameter setting
##############################################################################
learning_rate = 0.001
total_epoch = 30
batch_size = 16

# RNN 은 순서가 있는 자료를 다루므로, 한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지를 설정해야 함
# 이를 위해 가로 픽셀수를 n_input 으로, 세로 픽셀수를 입력 단계인 n_step 으로 설정
n_input = 28
n_step = 28
n_hidden = 64
n_class = 10
n_layers = 2
fc1_hidden = 32
fc2_hidden = 16


##############################################################################
# 신경망 모델 구성
##############################################################################
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Dropout_Rate1 = tf.placeholder(tf.float32)
Dropout_Rate2 = tf.placeholder(tf.float32)
TRAIN_BOOL = tf.placeholder(tf.bool)



# RNN 에 학습에 사용할 셀을 생성
# BasicRNNCell, BasicLSTMCell, GRUCell 들을 사용하면 다른 구조의 셀로 간단하게 변경 가능하며
# 본 코드에서는 gru를 이용하고, 과적합 방지를 위해 dropout을 적용해 주었음
def GRU_cell():
    cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell
def LSTM_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell
def RNN_cell():
    cell = tf.contrib.rnn.BasicRNNCell(num_units=n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell

with tf.variable_scope('GRUcell'):
    # 2개 층의 RNN 구조를 만들기 위해 MultiRNNCell을 이용
    multi_cells = tf.contrib.rnn.MultiRNNCell([GRU_cell() for _ in range(n_layers)])

    # RNN 신경망을 생성
    # 다음처럼 tf.nn.dynamic_rnn 함수를 사용하면 간단하게 RNN 신경망을 만들 수 있음
    # 원래는 초기 state를 지정해주고 번거로운 작업들이 많이 있었지만, 현재는 dynamic_rnn으로 쉽게 구성 가능
    outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

    # Fully-connected layer의 input을 맞춰줘야 함
    # 최종 y_pred를 Y의 형태인 [batch_size, n_class]로 맞춰줘야 함
    # outputs 의 형태를 이에 맞춰 변경해야함

    # outputs : [batch_size, n_step, n_hidden]
    #        -> [n_step, batch_size, n_hidden]
    outputs = tf.transpose(outputs, [1, 0, 2])
    #        -> [batch_size, n_hidden] # many to one 형태의 구조로 마지막의 hidden output만을 사용
    outputs = outputs[-1]


with tf.variable_scope('FC-layer1') as sc:
    FC1 = tf.contrib.layers.fully_connected(outputs, fc1_hidden, activation_fn=None)
    FC1_act = tf.nn.relu(tf.layers.batch_normalization(FC1, momentum=0.9, training=TRAIN_BOOL))

with tf.variable_scope('FC-layer2') as sc:
    FC2 = tf.contrib.layers.fully_connected(FC1_act, fc2_hidden, activation_fn=None)
    FC2_act = tf.nn.relu(tf.layers.batch_normalization(FC2, momentum=0.9, training=TRAIN_BOOL))
    y_pred = tf.contrib.layers.fully_connected(FC2_act, n_class, activation_fn=None)

with tf.variable_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))
    tf.summary.scalar(name='loss', tensor=cost)
    summary_op = tf.summary.merge_all()

with tf.variable_scope('optimizer'):
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)




##############################################################################
# 신경망 모델 학습
##############################################################################
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs/note11/extra01_MNIST_GRU_graph', sess.graph)
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0
    for i in range(total_batch):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # X 데이터를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input] 형태로 변환합니다.
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val, summary = sess.run([optimizer, cost, summary_op],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          Dropout_Rate1: 0.8,
                                          Dropout_Rate2: 0.2,
                                          TRAIN_BOOL: True})
        total_cost += cost_val
        if epoch*total_batch+i % 10 == 0:
            writer.add_summary(summary, epoch*total_batch+i)

    print('Epoch: {:02d}'.format(epoch+1),
          'Avg. cost: {:.4f}'.format(total_cost / total_batch))

print('===========Optimize complete===========')

# import os
# os.makedirs('./graphs/note11/extra01_MNIST_GRU_check')
saver = tf.train.Saver()
saver.save(sess, 'C:/Users/DeokseongSeo/Desktop/tensor_tmp/GRUmodel.ckpt')
# saver.restore(sess, './graphs/note11/extra01_MNIST_GRU_check/GRUmodel.ckpt')


##############################################################################
# 결과 확인
##############################################################################
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('Test Accuracy:', sess.run(accuracy,
                       feed_dict={X: test_xs,
                                  Y: test_ys,
                                  Dropout_Rate1: 1,
                                  Dropout_Rate2: 1,
                                  TRAIN_BOOL: False}))