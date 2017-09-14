#########################################################################
# module import, hyper-parameter 설정
#########################################################################
import tensorflow as tf
import os
from notes04_StructModel.ex04_1_structModel import *
from process_data import process_data



#########################################################################
# 모델을 구성하고, 데이터를 불러오고, 학습한 후 모형 저장
#########################################################################
# 저장용 폴더를 미리 만들어둬야 한다.
# os.makedirs('./graphs/notes04/skip_gram_checkpoints/')
# os.makedirs('./graphs/notes04/skip_gram_visualization/')

#skip gram model 객체 생성
model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
#model 내의 여러 정의할 부분을 하나의 메소드로 모두 실행
model.build_graph()

#http://mattmahoney.net/dc/ 에서 데이터 불러오기. (text8.zip)
#build_vocab 메소드에 의해 vocab_1000.tsv가 생성된다.
#이를 tensorboard를 켰을 때 embedding tab에서 load data에 넣어주면 임베딩된 결과물에 매칭해서 확인할 수 있다.
batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)

#model을 훈련시키고, 해당 parameter를 저장까지 한다.
# session을 생성하고 학습을 진행한다.
sess = tf.InteractiveSession()
# 학습한 Variable의 값을 저장하는 saver. 본 코드에서는 embed_matrix, nce_weight, nce_bias를 저장한다.
# 기본적으로는 모든 변수를 저장한다. 본 문제에서는 embed_matrix, nce_weight, nce_bias
saver = tf.train.Saver()
# 학습 진행
train_model(sess, model, batch_gen, NUM_TRAIN_STEPS)
# 모형 저장
saver.save(sess, './graphs/notes04/skip_gram_checkpoints/skipGram.ckpt')
sess.close()
