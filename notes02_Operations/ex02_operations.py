import tensorflow as tf
import numpy as np
import os

#맛보기
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

#with문으로 session을 열고 run(x)를 시행한 뒤 close를 한번에 해준다.
with tf.Session() as sess:
    # tensorboard를 실행하기 위해서는 FileWriter가 필요하다.
    writer = tf.summary.FileWriter('./graphs/notes02/warmingup', sess.graph)
    sess.run(x)
    writer.close()




###2. Constant types
#tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=false)
#1d tensor 생성(vector)
a = tf.constant([2, 2], name='vector')
#2x2 tensor 생성(matrix)
b = tf.constant([[0, 1], [2, 3]], name='b')

#모든 원소가 0인 텐서 생성
zeros_input = tf.zeros(shape=[2, 3], dtype=tf.int32)
tf.zeros_like(zeros_input)
#모든 원소가 1인 텐서 생성
ones_input = tf.ones(shape=[2, 3], dtype=tf.int32)
tf.ones_like(ones_input)
#모든 원소를 하나의 값으로 할당
const = tf.fill(dims=[2, 3], value=8)

#수열도 생성 가능. 그러나 non-iterable함
#tf.linspace(start, stop, num) #시작부터 끝까지 몇개로 쪼갤지 설정
tf.linspace(10.0, 13.0, 4, name='linspace') # 10.0, 11.0, 12.0, 13.0
#tf.range
tf.range(start=3, limit=18, delta=3) # 3, 6, 9, 12, 15
tf.range(start=3, limit=1, delta=-0.5) #3, 2.5, 2, 1.5
tf.range(start=0, limit=5) #0, 1, 2, 3, 4
#non-iterable
# for _ in np.linspace(0, 10 ,4): #OK
# for _ in tf.linspace(0, 10, 4): #TypeError
# for _ in range(4): #iterable
# for _ in tf.range(4): #non-iterable


###3. Math operations
a = tf.constant([3, 6])
b = tf.constant([2, 2])
tf.add(a, b) # 5, 8
tf.multiply(a, b) # 6, 12
tf.matmul(a, b) # error
tf.matmul(tf.reshape(a, shape=[1,2]), tf.reshape(b, shape=[2,1])) # 18. 모양을 벡터에서 매트릭스 형태로 바꿔준뒤 행렬 곱연산 가능
tf.div(a, b) # 1, 3. 몫
tf.mod(a, b) # 1, 0. 나머지


###4. Data types
#python native types
t = [[True, False, False],
       [False, False, True],
       [False, True, False]]
tf.zeros_like(t) # ==> 2x2 tensor
tf.ones_like(t) # ==> 2x2 tensor

#numpy data type도 가능
tf.ones([2,2], np.float32)


###5. Variables
a = tf.Variable(2, name='scalar')
b = tf.Variable([2,3], name='vector')
c = tf.Variable([[0, 1], [2, 3]], name='matrix')
W = tf.Variable(tf.zeros([784, 10]))
#init
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
#a, b만 init
init_ab = tf.variables_initializer([a, b], name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)
#W만 init
W = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)

#evaluate values of variables
W = tf.Variable(tf.truncated_normal([784, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W))

#Assign values to variables
W = tf.Variable(10)
assign_op = W.assign(1000) #assign 단계에서 사실 init을 진행하기 때문에, init할 필요 없음
#사실은 initial method가 assign의 형태를 띄고있음
with tf.Session() as sess:
    sess.run(assign_op)
    print(sess.run(W))

#U = W*2
W = tf.Variable(tf.truncated_normal([700, 10]))
#You should use this instead of the variable itself to initialize another variable with a value that depends on the value of this variable.
U = tf.Variable(W.initialized_value()*2) #W를 먼저 init 해줘야 한다.
init = tf.global_variables_initializer()
sess.run(init)
sess.run(U)


###6. Interactive Session - 당장 필요 없음
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
print(sess.run(c))
sess.close()


###7. control dependencies
#선행되어야 할 연산을 먼저 진행하도록 정의해줌
#Batch norm등을 이용할 때 사용하면 용이함
# with tf.Graph.control_dependencies([a, b, c]): #선행할 op들
#     d = ...
#     e = ... #실제로 실행하고자 하는 op들


###8. placeholders and feed_dict
#tf.placeholder(dtype, shape=None, name=None)
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a+b

#writer 이용해서 graph 저장하기
sess=tf.InteractiveSession()
writer = tf.summary.FileWriter('./graphs/notes02/ch8', sess.graph)
sess.run(c, feed_dict={a:[1,2,3]})
writer.close()


###9. The trap of lazy loading
#애초에 op 정의(권장됨)

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) #애초에 add op를 정의
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/notes02/ch9_1', sess.graph)
    for _ in range(10):
        sess.run(z)
    writer.close()

#사용할때마다 op 정의
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/notes02/ch9_2', sess.graph) #그래프를 저장한 뒤 tf.add가 정의되 있어서 그래프에 add node가 안보임
    for _ in range(10):
        sess.run(tf.add(x, y)) #create the op add only when you need to compute it
    writer.close()

