#Convolutional neural network
#MNIST data를 이용

import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

####################################################################################################
#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
####################################################################################################
#input: [Batch_size, Height, Width, Channels]
#filter: [Height, width, Input channels, Output channels]
#strides: 4차원의 1-d tensor로 표현하며, 각각의 방향으로 몇씩 움직일지를 의미한다. [1, 1, 1, 1] or [1, 2, 2, 1] 등. (1번째, 4번째는 1을 권고)
#padding: 'SAME' or 'VALID' (SAME가 패딩 있음, VALID가 패딩 없음을 의미)
#data_format: deault가 NHWC(위의 input 형태)

#tf.nn.에서 여러 conv2d를 제공하고 있다.
#tf.nn.conv2d
#tf.nn.depthwise_conv2d
#tf.nn.separable_conv2d
#tf.nn.atrous_conv2d
#tf.nn.conv2d_transpose
#등등... 필요한 conv를 공부해서 적절하게 사용하자.




""" 
MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""


# Step 1: Read in data
# TF의 built in function을 이용해서 MNIST를 load해서 ./data/mnist 폴더에 저장하고, one_hot으로 label을 표현
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)



# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1
N_CLASSES = 10



# Step 3: create placeholders for features and labels
# MNIST의 각 이미지는 28*28 = 784 차원
# tensor로 표현하자면 1x784가 된다.
# 각 hidden layer마다 dropout을 적용하기 위해 placeholder를 선언한다.
# placeholder의 첫번째 크기부분을 None으로 주어서 batch_size를 변경해가며 실험할 수 있다.
with tf.name_scope('data'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X_placeholder')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y_placeholder')
# dropout의 keepprob 정의
dropout = tf.placeholder(dtype=tf.float32, name='dropout')
# global_step 정의
global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')



# Step 4 + 5: create weights + do inference
# model : [conv, relu] -> pool -> [conv, relu] -> pool -> fully connected -> softmax
# conv layer안에 relu 함수를 넣어서 같이 처리해줌

#variable_scope를 통해서 각 layer를 정의하고 with문 안에 여러 variable들을 정의한다.
#이렇게 되면 서로 다른 층끼리 name이 겹칠일이 거의 없어진다. 예)conv1의 weight 변수: conv1-weight, conv2의 weight 변수: conv2-weight
#variable_scope에서는 tf.Variable대신 tf.get_variable로 변수를 선언한다.
#tf.get_variable(name, shape, initializer)

#CONV1 LAYER
with tf.variable_scope('conv1') as scope:
    # input : BATCH_SIZE * 28 * 28 * 1
    #tf.nn.conv2d에 적용하기 위해 input으로 들어온 image를 [BATCH_SIZE, 28, 28, 1]로 모양을 다시 잡아준다.
    images = tf.reshape(X, shape=[-1, 28, 28, 1]) #BATCH_SIZE만큼 feed_dict를 통해 들어오는데([BATCH_SIZE, 784]), 그것을 [?, 28, 28, 1]로 맞춰준다.
    kernel = tf.get_variable(name='kernel',                                     #kernel이라는 이름(즉, conv1-kernel이 된다.)
                             shape=[5, 5, 1, 32],                               #5*5*1의 필터를 32개 만들어서 다음 layer에서는 32층이 된다.
                             initializer=tf.truncated_normal_initializer())     #초기화는 normal로 적용. truncated_normal_initializer는 클래스이다.
    biases = tf.get_variable(name='biases',                                     #biases라는 이름(즉, conv1-biases가 된다.)
                             shape=[32],                                        #다음 layer의 32층 각각에 대한 bias를 정의
                             initializer=tf.random_normal_initializer())        #초기화
    conv = tf.nn.conv2d(input=images,           #input에 위에서 모양을 잡아준 images를 줌
                        filter=kernel,          #filter로 위에서 정의한 kernel 적용
                        strides=[1, 1, 1, 1],   #모든 방향으로 1씩 움직이면서 학습. 아마도 하나의 kernel을 [ , 1, 1, ]로 움직이는데, 맨앞의 1은 batch 내의 데이터를 1개씩 모두 훑는것이고, 맨뒤의 1은 데이터의 channel을 모두 훑는 것을 의미
                        padding='SAME')         #SAME 패딩을 줌으로써 output의 height, width도 28, 28로 다시 만들어준다.
    conv1 = tf.nn.relu(conv + biases, name=scope.name) #최종적으로 이번 variable_scope의 마무리를 짓는 층이다. relu안에 위에서 filter를 거친 conv와 bias를 더해준다.
    # output : BATCH_SIZE * 28 * 28 * 32

#POOL1 LAYER
with tf.variable_scope('pool1') as scope:
    # input : BATCH_SIZE * 28 * 28 * 32
    pool1 = tf.nn.max_pool(value=conv1,             #conv1 객체에 대해 max pooling 진행
                           ksize=[1, 2, 2, 1],      #[1, 2, 2, 1]만큼을 보고 그중 최대값으로 대체
                           strides=[1, 2, 2, 1],    #[1, 2, 2, 1]만큼 이동
                           padding='SAME')          #패딩을 SAME으로 주어 output 크기 같게 해준다.
    # output : BATCH_SIZE * 14 * 14 * 32

#CONV2 LAYER
with tf.variable_scope('conv2') as scope:
    # input : BATCH_SIZE * 14 * 14 * 32
    kernel = tf.get_variable(name='kernels',
                             shape=[5, 5, 32, 64],  #kernel 크기가 input의 channel이 32, output하는 channel이 64로 되었다. H, W는 여전히 5*5이다.
                             initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable(name='biases',
                             shape=[64],
                             initializer=tf.truncated_normal_initializer())
    conv = tf.nn.conv2d(input=pool1,
                        filter=kernel,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)
    # output : BATCH_SIZE * 14 * 14 * 64

#POOL2 LAYER
with tf.variable_scope('pool2') as scope:
    # input : BATCH_SIZE * 14 * 14 * 64
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # output : BATCH_SIZE * 7 * 7 * 64

#FULLY CONNECTED LAYER
with tf.variable_scope('fc') as scope:
    # input : BATCH_SIZE * 14 * 14 * 64
    # 이전의 pool2 layer에서 7*7*64가 BATCH_SIZE 개수만큼 온다. 이것을 1024 차원으로 먼저 보낸다.
    input_features = 7 * 7 * 64 #3136
    #weight 정의
    w = tf.get_variable(name='weights',
                        shape=[input_features, 1024],
                        initializer=tf.truncated_normal_initializer())
    #bias 정의
    b = tf.get_variable(name='biases',
                        shape=[1024],
                        initializer=tf.truncated_normal_initializer())
    #pool2를 쭉 펼쳐서 BATCH_SIZE * (7*7*64)로 길게 늘인다. 2차원 텐서 형태
    pool2 = tf.reshape(pool2, [-1, input_features])
    #fully connected layer 거친 후 relu 까지 적용
    fc = tf.nn.relu(features=tf.matmul(pool2, w) + b, name='relu')
    #다시 dropout까지 적용
    fc = tf.nn.dropout(x=fc, keep_prob=dropout, name='relu_dropout') #아래에서 fc에 대해 dropout(0.75로 설정)만큼 유지하고 나머지 노드는 꺼준다.
    # output : BATCH_SIZE * 1024

#SOFTMAX_LINEAR LAYER
with tf.variable_scope('softmax_linear') as scope:
    # input : BATCH_SIZE * 1024
    #weight
    w = tf.get_variable(name='wieght',
                        shape=[1024, N_CLASSES],
                        initializer=tf.truncated_normal_initializer()) #1024 차원에서 N_CLASSES(=10)차원으로 보낸다.
    #bias
    b = tf.get_variable(name='biases',
                        shape=[N_CLASSES],
                        initializer=tf.truncated_normal_initializer())
    logits = tf.matmul(fc, w) + b
    # input : BATCH_SIZE * 10



# Step 6: define loss function
# softmax cross entropy with logits를 이용해서 loss를 정의한다.
# 구한 cross entropy를 평균낸다.
with tf.name_scope('loss'):
    #loss
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss = tf.reduce_mean(input_tensor=entropy, name='loss')
    #tensorboard에 그릴 summary 정의
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('histogram_loss', loss)
    summary_op = tf.summary.merge_all()



# Step 7: define training op
# gradient descent 방법으로 loss를 최소화하는 방향으로 업데이트 한다. 사용한 방법은 Adam
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss, global_step=global_step)





###########################################################################################
# Let's Train!!!
###########################################################################################
# parameter 저장용 폴더 생성해두기
# os.makedirs('./graphs/notes07/mnist_checkpoint/')
with tf.Session() as sess:
    #변수 초기화
    sess.run(tf.global_variables_initializer())
    #파라미터 저장 객체 생성. 최대 3개만 저장
    saver = tf.train.Saver(max_to_keep=3)
    #tensorboard 시각화용 writer 객체 생성
    writer = tf.summary.FileWriter('./graphs/notes07/mnist_graph', sess.graph)

    #초기 step은 global_step.eval로 할당
    initial_step = global_step.eval()

    #학습 시간을 보기 위해 start 시간을 찍어준다.
    start_time = time.time()
    #전체 데이터 개수를 BATCH_SIZE로 나누어서 몇번 돌아야 한번의 epoch인지를 n_batches로 저장
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    #loss 계산
    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS): #배치단위로 n_batches * N_EPOCHS 이만큼 돌면서 학습
        #배치 생성
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        #최적화 진행
        _, loss_batch, tr_summary = sess.run([optimizer, loss, summary_op],
                                 feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        total_loss += loss_batch
        if (index+1) % SKIP_STEP == 0:
            # SKIP_STEP 마다 loss가 제대로 줄고 있는지 확인한다.
            print('Average loss at step {}: {:5.1f}'.format(index+1, total_loss/SKIP_STEP))
            # 다시 다음번 SKIP_STEP을 위해 total_loss를 초기화 해둔다.
            total_loss = 0.0
            writer.add_summary(tr_summary, global_step=index)
            # 이렇게 자주 저장 하는 경우에는 속도가 많이 늦어지게 된다.
            # 따라서 제일 마지막에 한번 저장해 주는 방식을 추천하는데, 지금의 경우 공부를 위해, 최대 3개까지 저장되는 것을 보기 위한 코드로 진행
            saver.save(sess, './graphs/notes07/mnist_checkpoint/convNet_{}.ckpt'.format(index+1))

    #학습 완료
    print("Optimization Finished!")  # 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))
    writer.close()


    # test the model
    # test data에 대한 성능을 살펴볼 배치 개수 결정
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        # test 배치 할당
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        # test 데이터를 모형에 적용
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT}) #test시에는 굳이 optimizer 안해도 되긴 함. 더 많은 데이터를 사용해서 일단 weight update 진행
        #적용 후 얻은 logits_batch에 softmax를 적용해서 preds를 구함.(확률값을 얻을 수 있음)
        preds = tf.nn.softmax(logits_batch)
        #Y_batch, preds에 대해 최대값을 갖는 위치를 가지고 와서 얼마나 많이 맞는지 배치단위로 확인한다.
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    #전체 개수로 맞은 개수를 나눠서 최종 accuracy를 구함
    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples)) #Accuracy 0.9171

