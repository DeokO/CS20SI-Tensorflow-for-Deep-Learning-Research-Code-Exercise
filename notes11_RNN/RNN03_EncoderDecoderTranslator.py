# **여러 길이를 가진 단어들에 대해 영어 단어를 한국어 단어로 번역하는 코드

# 이 코드는 RNN에 대해 익히기 위해 golbin님의 git에있는 코드와 Danijar Hafner님의 자료를 참고해서 응용
# https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/03%20-%20Seq2Seq.py
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq을 구현해 봅니다.

# golbin님 코드로부터 변경된 사항
# - dynamic_rnn에서 sequence_length를 사용하기 위해 예제 단어 추가(길이가 다른 input을 dynamic하게 처리)
# - LSTM cell을 이용하고, RNN을 2층으로 쌓고 Dropout을 적용
#   (데이터가 인공으로 워낙 작게 되있으니 1층만 했지만, hyper-parameter에서 1을 n으로 변경해서 층을 추가할 수 있음)
#   (마찬가지 이유로 FC-layer도 1층만 쌓음)
##############################################################################
# import module
##############################################################################
import tensorflow as tf
import numpy as np



##############################################################################
# input define
##############################################################################
# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']

# S, E, P와 알파벳 소문자들, 그리고 해당하는 한글 단어를 가능한 범위로 정의함
# 모든 경우를 포괄하기 위해서는 한글 단어에 대한 모든 단어를 다 주던지, 아니면 자소단위로 접근하면 될 것
char_arr = np.unique([c for c in 'abcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑열쇠아이폰무거운내이름'])
char_arr = np.r_[char_arr, ['S', 'E', 'P']]
# 이를 dictionary형태로 저장해 둠
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑'],
            ['key', '열쇠'], ['iphone', '아이폰'],
            ['heavy', '무거운'], ['sds', '내이름']]

# 결국 데이터 구조는 아래 batch를 만드는 함수를 통해
# input: g, a, m, e
# output: S, 놀, 이
# target: 놀, 이, E
# 형태로 모든 seq_data를 만들어 준다.
# 그런데 위와 같이 글자 자체로 저장하는 것이 아닌, one-hot-encoding으로 저장해둔다.
def make_batch(seq_data):
    enc_batch = []
    dec_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 위에서 정의한 num_dic를 이용해서 integer 배열로 만든다.
        enc_input = [num_dic[n] for n in seq[0]]
        # 인코더 input의 길이를 최대 개수인 6개(n_step)로 맞춰주고, 나중에 dynamic_rnn에서 개수를 전달하여 제거
        enc_diff = enc_step - len(enc_input)
        enc_input.extend(np.repeat(dic_len, enc_diff)) #padding용으로 제일 마지막 숫자를 일단 넣어두고, 아래 줄에서 처리해줌
        enc_batch.append(np.r_[np.eye(dic_len), np.repeat(0, dic_len).reshape([1, -1])][enc_input])  # 패딩을 고려한 input 생성

        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        dec_input = [num_dic[n] for n in ('S' + seq[1])]
        # 디코더 input의 길이를 최대 개수인 4+1개(n_step)로 맞춰주고(S, 1개 추가), 나중에 dynamic_rnn에서 개수를 전달하여 제거
        dec_diff = (dec_step+1) - len(dec_input)
        dec_input.extend(np.repeat(dic_len, dec_diff))
        dec_batch.append(np.r_[np.eye(dic_len), np.repeat(0, dic_len).reshape([1, -1])][dec_input])

        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]
        target_diff = (dec_step+1) - len(target)
        target.extend(np.repeat(0, target_diff))
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return enc_batch, dec_batch, target_batch

# dynamic_rnn에서 sequence_length 옵션에 전달할 각 문서의 길이를 산출하는 함수
def length(batch_x):
    # 모두 양수로 만들고, 패딩이 아닌 것은 1로, 패딩은 0으로 되도록 max를 취함
    _max = np.max(np.abs(batch_x), 2)
    # 각 batch별로 sum을 해서 해당 input의 길이가 몇인지 산출
    leng = np.sum(_max, 1)
    return leng



##############################################################################
# Hyper-parameter setting
##############################################################################
learning_rate = 0.0005
n_hidden = 128
total_epoch = 1000
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len

# fully connected layer의 hidden node 수 설정
fc1_hidden = 256

#encoder와 decoder의 input max step수 산출
enc_step = np.max(list(map(lambda x: len(x[0]), seq_data)))
# 'S'에 해당하는 한개가 무조건 있으므로 max에도 1을 더해줌
dec_step = np.max(list(map(lambda x: len(x[1]), seq_data)))




##############################################################################
# 신경망 모델 구성
##############################################################################
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
ENC_INPUT = tf.placeholder(tf.float32, [None, None, dic_len])
DEC_INPUT = tf.placeholder(tf.float32, [None, None, dic_len])
# dynamic_rnn에서 sequential length를 이용하기 위한 placeholder
ENC_SEQ = tf.placeholder(tf.int32, [None])
DEC_SEQ = tf.placeholder(tf.int32, [None])
# [batch size, time steps]
TARGETS = tf.placeholder(tf.int64, [None, None])
# dropout을 위한 placeholder
Dropout_Rate1 = tf.placeholder(tf.float32) #input keep prob
Dropout_Rate2 = tf.placeholder(tf.float32) #output keep prob
# Batch-normalization에서 train 여부를 위한 placeholder
TRAIN_BOOL = tf.placeholder(tf.bool)

# 인코더 셀을 구성한다.
# 인코더의 input으로 ENC_INPUT: input_batch가 나중에 feed_dict를 통해 들어오고,
# 마지막 코드에서 enc_states를 디코더 셀에 전달해 줘야 하는데, 이부분이 중요
with tf.variable_scope('Encode'):
    enc_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)

    #인코더 셀의 마지막 enc_states를 디코더 셀의 initial_state로 전달하기 위해 저장
    enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, ENC_INPUT, sequence_length=ENC_SEQ, dtype=tf.float32)

# 디코더 셀을 구성한다.
# 디코더의 input으로 DEC_INPUT: output_batch가 나중에 feed_dict를 통해 들어오고,
# 인코더로부터 얻은 enc_states를 initial_state에 전달해줌
with tf.variable_scope('Decode'):
    dec_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=Dropout_Rate1, output_keep_prob=Dropout_Rate2)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심
    dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, DEC_INPUT, initial_state=enc_states, sequence_length=DEC_SEQ, dtype=tf.float32)

# 디코더의 output(dec_outputs)에 대해 모양을 flatten하고,
# DEC_SEQ에 의해 의미없는 부분을 제거한 뒤 FC-layer를 거쳐 이후 역전파등을 위한 준비를 함
with tf.variable_scope('FC-preprocessing'):
    #디코더의 최종 output 중에서 DEC_SEQ 이후의 0벡터들을 제거
    flat = tf.reshape(dec_outputs, [-1, n_hidden])
    row_wise_sum = tf.reduce_sum(tf.abs(flat), 1)
    selected_nonzero = tf.not_equal(row_wise_sum, 0)
    rnn_output_flat = tf.boolean_mask(tensor=flat, mask=selected_nonzero)

    # target에서도 마찬가지 부분을 제거
    target_flat = tf.reshape(TARGETS, [-1, 1])
    removed_target = tf.reshape(tf.boolean_mask(tensor=target_flat, mask=selected_nonzero), [-1])

# Fully-connected layer 1 & y_pred
# 디코더에서 얻은 dec_outputs에 Fully connected layer를 연결해주고 배치놈을 적용한 후 relu를 적용
# FC1_act에 Fully connected layer를 연결해주어 최종 logit을 만듦
with tf.variable_scope('FC-layer1'):
    FC1 = tf.contrib.layers.fully_connected(rnn_output_flat, fc1_hidden, activation_fn=None)
    FC1_act = tf.nn.relu(tf.layers.batch_normalization(FC1, momentum=0.9, training=TRAIN_BOOL))
    y_pred = tf.contrib.layers.fully_connected(FC1_act, n_class, activation_fn=None)

# Define Loss
with tf.variable_scope('loss'):
    #sparse_softmax 를 이용해서 integer 형태의 두 값으로 loss를 구함
    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_pred, labels=removed_target))
    # tensorboard로 볼 loss 정의
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
writer = tf.summary.FileWriter('./graphs/notes11/RNN03_EncoderDecoderTranslator_graph', sess.graph)

input_batch, output_batch, target_batch = make_batch(seq_data)
for epoch in range(total_epoch):
    ind = np.random.choice(range(len(seq_data)), np.int(len(seq_data) * 0.7), replace=False)
    batch_enc = np.take(input_batch, ind, axis=0)
    batch_dec = np.take(output_batch, ind, axis=0)
    batch_target = np.take(target_batch, ind, axis=0)
    enc_len = length(batch_enc)
    dec_len = length(batch_dec)

    _, loss, summary = sess.run([optimizer, cost, summary_op],
                       feed_dict={ENC_INPUT: batch_enc,
                                  DEC_INPUT: batch_dec,
                                  TARGETS: batch_target,
                                  Dropout_Rate1: 0.5,
                                  Dropout_Rate2: 0.5,
                                  ENC_SEQ: enc_len,
                                  DEC_SEQ: dec_len,
                                  TRAIN_BOOL: True})

    # writer에 10단위로 한번씩 stack
    if epoch % 10 == 0:
        writer.add_summary(summary, epoch)

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

# 학습한 parameter(Weights, Biases)를 saver를 통해 저장
# import os
# os.makedirs('./graphs/notes11/RNN03_EncoderDecoderTranslator_check')
saver = tf.train.Saver()
saver.save(sess, './graphs/notes11/RNN03_EncoderDecoderTranslator_check/dynamic_GRU_EncoderDecoder_translator.ckpt')

# 나중에 불러오고 싶을때는 restore를 이용
# saver.restore(sess, './graphs/notes11/RNN03_EncoderDecoderTranslator_check/dynamic_GRU_EncoderDecoder_translator.ckpt')



##############################################################################
# 결과 확인 (번역 테스트)
##############################################################################
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P(padding) 으로 채운다.
    # 예: ['word', 'PPPP'], ['heavy', 'PPPP']
    seq_data = [word, 'P' * dec_step]

    # 입력받은 seq_data에 대해 학습때와 같이 데이터셋을 구성해줌
    input_batch, output_batch, target_batch = make_batch([seq_data])
    enc_len = length(input_batch)
    dec_len = length(output_batch)

    # 결과가 [batch size * time step, input] 으로 나오기 때문에, (이부분은 golbin님것과 다른 부분인데, dynamic을 적용하기 위해 flatten했기 때문에 차원이 다름)
    # 1번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(y_pred, 1)

    result = sess.run(prediction,
                      feed_dict={ENC_INPUT: input_batch,
                                  DEC_INPUT: output_batch,
                                  TARGETS: target_batch,
                                  Dropout_Rate1: 1,
                                  Dropout_Rate2: 1,
                                  ENC_SEQ: enc_len,
                                  DEC_SEQ: dec_len,
                                  TRAIN_BOOL: False})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated


print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wwrd ->', translate('wwrd'))
print('key ->', translate('key'))
print('kei ->', translate('kep'))
print('iphone ->', translate('iphone'))
print('ipone ->', translate('ipone'))
print('sds ->', translate('sds'))
print('sbs ->', translate('sbs'))
