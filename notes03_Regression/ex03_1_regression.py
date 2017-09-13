import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd #엑셀 리딩 모듈


#####################################################################################
# 1차식으로 fitting
#####################################################################################
# step1: .xls 파일 읽어오기
DATA_FILE = "./data/fire_theft.xls"
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# step2 : input X, label Y를 위한 placeholder 선언
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# step3 : weight, bias 0으로 초기화해서 생성
w = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')

# step4 : model
Y_predicted = X * w + b

# step5 : loss
loss = tf.square(Y - Y_predicted, name='loss')

# step6 : Gradient descent 이용해서 학습하기 위한 최적화 객체 선언
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

with tf.Session() as sess:
    #step7 : Variables 초기화
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/notes03/regression1', sess.graph)
    # Step 8: train the model
    for i in range(100):  # train the model 100 times
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    #step9 : 최종 출력
    w_value, b_value = sess.run([w, b])

#plotting
plt.scatter(data[:,0], data[:,1], c='b', label='Real data')
t = np.arange(0., 40., 0.2)
plt.plot(t, w_value*t+b_value, 'r-', label='Predicted data')
plt.legend()
plt.show()






#####################################################################################
# 2차식으로 fitting
#####################################################################################
# step1: .xls 파일 읽어오기
DATA_FILE = "./data/fire_theft.xls"
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
x = data[:,[0]]
y = data[:,[1]]
print(x.shape, y.shape)

# step2 : input X, label Y를 위한 placeholder 선언
X = tf.placeholder(tf.float32, shape=[None, 1], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

#step3 : w1, w2, bias
w = tf.Variable(initial_value=tf.random_normal(shape=[1]), name="weights_1")
u = tf.Variable(initial_value=tf.random_normal(shape=[1]), name="weights_2")
b = tf.Variable(initial_value=tf.random_normal(shape=[1]), name="bias")

#step4 : predict y
Y_predicted = X*X*w + X*u + b

#step5 : train
loss = tf.reduce_mean(tf.square(Y - Y_predicted))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss) #Adam optimizer를 사용해본다.
with tf.Session() as sess:
    #step7 : Variables 초기화
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/notes03/regression2', sess.graph)

    #step8 : train
    for i in range(100000): #100000 epochs - 더 오래 돌리고 괜찮은 optimizer, learning rate를 이용하면 더 좋은 결과를 얻을 수 있음
        los, _ = sess.run([loss, optimizer], feed_dict={X: x,  Y: y})
        print(i, los)

    #step9 : 최종 출력
    w_value, u_value, b_value = sess.run([w, u, b])

#plotting
plt.scatter(data[:,0], data[:,1], c='b', label='Real data')
t = np.arange(0., 40., 0.2)
plt.plot(t, w_value*(t**2) + u_value*t +b_value, 'r-', label='Predicted data')
plt.legend()
plt.show()




#인공 data로 잘 적합되었는지 확인하기
X_input = np.linspace(-1, 1, 100)
Y_input = X_input * 3 + np.random.randn(X_input.shape[0]) * 0.5

X = tf.placeholder(dtype=tf.float32, name='X')
Y = tf.placeholder(dtype=tf.float32, name='Y')

w = tf.Variable(initial_value=0.01, name='weight')
b = tf.Variable(initial_value=0.0, name='bias')

Y_predicted = w*X + b
loss = tf.square(Y-Y_predicted, name='loss')
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/notes03/regression3', sess.graph)
    for i in range(10): #epoch 10
        for x, y in np.c_[X_input, Y_input]:
            los, _ = sess.run([loss, opt], feed_dict={X: x, Y: y})
        print("Epoch {} - loss : {}".format(i, los))
    w_value, b_value = sess.run([w, b])

#plot
t = np.linspace(-1, 1, 100)
plt.plot(t, w_value*t+b_value, c='g', label='predicted_line')
plt.scatter(X_input, Y_input, c='r', label='training_data')
plt.legend()
plt.show()












###Optimizer
#gradient descent 방법을 위에서는 사용했는데 이뿐만아니라 여러 방법들이 있다.
#loss, cost에 종속적인 Variable들이 모두 학습이 되며, 학습을 원치 않을 경우에는 trainable=False를 주면 된다.
#예를들어, global step같은 변수는 학습하지 않는 용도로 사용하는 것을 많이 볼 수 있을 것이다.
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32) #step이 커질수록 점점 작은 learning rate를 적용하게 됨
increment_step = global_step.assign_add(1) #global_step + 1값을 increment_step에 저장
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #learning_rate 자체가 tensor라 생각 할 수 있다.

#tf.Variable을 자세히 살펴보자
#tf.Variable(initial_value=None, trainable=True, collections=None,
#            validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None,
#            expected_shape=None, import_scope=None)

#optimizer로 미분 계산하는 것도 살펴볼 수 있는데, 그것까지는 공부할 필요는 없을 듯 하다.
#특정 변수만 update 할때 사용 가능하다.
# # create an optimizer.
# optimizer = GradientDescentOptimizer(learning_rate=0.1)
# # compute the gradients for a list of variables.
# grads_and_vars = opt.compute_gradients(loss, <list of variables>)
# # grads_and_vars is a list of tuples (gradient, variable). Do whatever you
# # need to the 'gradient' part, for example, subtract each of them by 1.
# subtracted_grads_and_vars = [(gv[0] - 1.0, gv[1]) for gv in grads_and_vars]
# # ask the optimizer to apply the subtracted gradients.
# optimizer.apply_gradients(subtracted_grads_and_vars)

#사용 가능한 optimizer. AdamOptimizer를 사용하는 것을 요즘 권고함.(2017.)
# tf.train.GradientDescentOptimizer
# tf.train.AdadeltaOptimizer
# tf.train.AdagradOptimizer
# tf.train.AdagradDAOptimizer
# tf.train.MomentumOptimizer
# tf.train.AdamOptimizer
# tf.train.FtrlOptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSPropOptimizer



