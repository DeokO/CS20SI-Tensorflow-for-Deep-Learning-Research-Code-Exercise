# 이 코드는 RNN에 대해 익히기 위해 golbin님의 git에있는 코드를 참조해서 응용
# https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/02%20-%20Autocomplete.py
# 4개의 글자를 가진 단어를 학습시켜, 3글자만 주어지면 나머지 한 글자를 추천하여 단어를 완성하는 프로그램입니다.
# golbin님 코드로부터 변경된 사항
# - dynamic_rnn에서 sequence_length를 사용하기 위해 예제 단어 추가(길이가 다른 input을 dynamic하게 처리)
# - LSTM cell을 이용하고, RNN을 2층으로 쌓고 Dropout을 적용


##############################################################################
# import module
##############################################################################
import tensorflow as tf
import numpy as np



##############################################################################
# basic setting
##############################################################################
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# one-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 생성
# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
# 사용할 dictionary의 개수
# 현재는 character 단위의 input이므로, character의 개수로 봐도 됨
dic_len = len(num_dic)

# 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것
# wor -> X,     d -> Y
# iphon -> X,   e -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind',
            'key', 'train', 'zon', 'apple', 'iphone', 'pen', 'ear', 'eye', 'lib', 'heavy',
            'usb', 'sun', 'spray', 'sds']

# batch 단위로 들어갈 X(one-hot), Y(int) 데이터를 생성
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 여기서 생성하는 input_batch 와 target_batch는 알파벳 배열의 인덱스 번호
        # input: [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
        input = [num_dic[n] for n in seq[:-1]]
        # input의 길이를 최대 개수인 6개(n_step)로 맞춰주고, 나중에 dynamic_rnn에서 개수를 전달하여 제거
        diff = n_step - len(input)
        input.extend(np.repeat(1, diff))
        # target: 3, 3, 15, 4, 3 ...
        target = num_dic[seq[-1]]

        # input에 대해서는 one-hot 인코딩을 진행
        # if input is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  .  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        input_batch.append(np.eye(dic_len)[input])

        # 지금까지 손실함수로 사용하던 softmax_cross_entropy_with_logits 함수는 label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
        # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits는 one-hot 인코딩을 사용하지 않으므로
        # index 를 그냥 넘겨주면 됨
        # target class의 개수가 여러개인 경우, 유용하게 사용될 수 있음 (예: 기계번역 등)
        target_batch.append(target)

    return input_batch, target_batch



##############################################################################
# Hyper-parameter setting
##############################################################################
learning_rate = 0.001
n_hidden = 128
total_iter = 1000

# 타입 스텝: [1 2 3 4 5 6] => 6
# RNN 을 구성하는 시퀀스의 갯수. cell이 펼쳐졌을때의 개수(최대값 제공)
n_step = np.max(list(map(lambda x: len(x), seq_data)))

# RNN의 layer
n_layers = 2

# 입력값 크기. 알파벳에 대한 one-hot 인코딩이므로 26개
# 예) c => [0 0 1 0 0 0 0 0 0 0 0 ... 0]
# 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류
n_input = n_class = dic_len
fc1_hidden = 32
fc2_hidden = 16



##############################################################################
# 신경망 모델 구성
##############################################################################
# input
X = tf.placeholder(tf.float32, [None, None, n_input])
# output
# 비용함수에 sparse_softmax_cross_entropy_with_logits 을 사용하므로
# loss 계산을 위한 출력값(Y_pred)과 원본값(Y)의 형태는 one-hot vector가 아니라 인덱스 숫자를 그대로 사용
# 다음처럼 하나의 값만 있는 1차원 배열을 입력값으로 받음
# [3] [3] [15] [4] ...
# 기존처럼 one-hot 인코딩을 사용한다면 입력값의 형태는 [None, n_class]가 맞음
Y = tf.placeholder(tf.int32, [None])
# dynamic_rnn에서 sequential length를 이용하기 위한 placeholder
SEQ = tf.placeholder(tf.int32, [None])
# dropout을 위한 placeholder
Dropout_Rate1 = tf.placeholder(tf.float32)
Dropout_Rate2 = tf.placeholder(tf.float32)
# Batch-normalization에서 train 여부를 위한 placeholder
TRAIN_BOOL = tf.placeholder(tf.bool)



# RNN 에 학습에 사용할 셀을 생성
# BasicRNNCell, BasicLSTMCell, GRUCell 들을 사용하면 다른 구조의 셀로 간단하게 변경 가능하며
# 본 코드에서는 LSTM을 이용하고, 과적합 방지를 위해 dropout을 적용해 주었음
def GRU_cell(num_units):
    cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell
def LSTM_cell(num_units):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell
def RNN_cell(num_units):
    cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)
    return cell

with tf.variable_scope('LSTMcell'):
    # 여러개의 셀을 조합한 RNN 셀을 생성합니다.
    multi_cells = tf.contrib.rnn.MultiRNNCell([LSTM_cell(num_units=n_hidden) for _ in range(n_layers)])

    # RNN 신경망을 생성
    outputs, _states = tf.nn.dynamic_rnn(cell=multi_cells, inputs=X, sequence_length=SEQ, dtype=tf.float32)

    # Fully-connected layer의 input을 맞춰줘야 함
    outputs_tr = tf.transpose(outputs, [1, 0, 2])
    outputs_last = outputs_tr[-1]

# Fully-connected layer 1
with tf.variable_scope('FC-layer1') as sc:
    FC1 = tf.contrib.layers.fully_connected(outputs_last, fc1_hidden, activation_fn=None)
    FC1_act = tf.nn.relu(tf.layers.batch_normalization(FC1, momentum=0.9, training=TRAIN_BOOL))

# Fully-connected layer 2
with tf.variable_scope('FC-layer2') as sc:
    FC2 = tf.contrib.layers.fully_connected(FC1_act, fc2_hidden, activation_fn=None)
    FC2_act = tf.nn.relu(tf.layers.batch_normalization(FC2, momentum=0.9, training=TRAIN_BOOL))
    y_pred = tf.contrib.layers.fully_connected(FC2_act, n_class, activation_fn=None)

# Define Loss
with tf.variable_scope('loss'):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))
    tf.summary.scalar(name='loss', tensor=cost)
    summary_op = tf.summary.merge_all()

# Parameter update와 동시에 batch normalization의 parameter도 update되도록 control_dependencies를 걸어줌
with tf.variable_scope('optimizer'):
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



##############################################################################
# 신경망 모델 학습
##############################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# graph 확인과 loss 확인을 위해 summary를 저장할 writer 생성
writer = tf.summary.FileWriter('./graphs/notes11/RNN02_word_Autocomplete_graph', sess.graph)

# input data를 산출
input_batch, target_batch = make_batch(seq_data)
# 각 input의 길이를 미리 계산해둠
seq_len = list(map(lambda x: x.shape[0], input_batch))

# total_iter 만큼 training
for epoch in range(total_iter):

    #기존 코드에서는 같은 data가 계속 feed_dict 되는데,
    #학습 데이터를 random하게 넣어주는 장치를 걸어 줌
    ind = np.random.choice(range(len(seq_data)), np.int(len(seq_data) * 0.7), replace=False)

    _, loss, summary = sess.run([optimizer, cost, summary_op],
                       feed_dict={X: np.take(input_batch, ind, axis=0), # input X
                                  Y: np.take(target_batch, ind),        # output Y
                                  SEQ: np.take(seq_len, ind),           # sequence_length SEQ
                                  Dropout_Rate1: 0.8,
                                  Dropout_Rate2: 0.2,
                                  TRAIN_BOOL: True})

    # writer에 10단위로 한번씩 stack
    if epoch % 10 == 0:
        writer.add_summary(summary, epoch)

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

# 학습한 parameter(Weights, Biases)를 saver를 통해 저장
# import os
# os.makedirs('./graphs/notes11/RNN02_word_Autocomplete_ckeck')
saver = tf.train.Saver()
saver.save(sess, './graphs/notes11/RNN02_word_Autocomplete_ckeck/LSTMmodel.ckpt')

# 나중에 불러오고 싶을때는 restore를 이용
# saver.restore(sess, './graphs/notes11/RNN02_word_Autocomplete_ckeck/LSTMmodel.ckpt')



##############################################################################
# 결과 확인
##############################################################################
# 레이블값이 정수이므로 예측값도 정수로 변경
prediction = tf.cast(tf.argmax(y_pred, 1), tf.int32)
# one-hot 인코딩이 아니므로 입력값을 그대로 비교
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

# 예측값과 정밀도값을 측정
# training data에 대해서 파악하는 것이므로 큰 의미는 없지만,
# 잘 학습되는지 확인
input_batch, target_batch = make_batch(seq_data)
predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch,
                                            Y: target_batch,
                                            SEQ: seq_len,
                                            Dropout_Rate1: 1,
                                            Dropout_Rate2: 1,
                                            TRAIN_BOOL: False})

# 예측한 단어가 어떤지 확인해보기
predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:-1] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [val[:-1] + ' ' for val in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)

