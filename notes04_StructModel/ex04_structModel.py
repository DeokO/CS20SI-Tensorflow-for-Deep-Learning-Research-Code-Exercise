###word2vec 구축 연습
#Phase 1: assemble your graph
# - define placeholders, weights, inference model, loss, optimizer
#Phase 2: execute the computation
# - initialize all model variables for the first time
# - feed training data
# - execute the inference model
# - compute cost
# - adjust the model parameters


# ###################################################### 이부분은 line by line 으로 연습하는 부분. 아래에 full code가 있음
# # 필요한 라이브러리들을 임포트
# import tensorflow as tf
#
#
#
# VOCAB_SIZE = 50000
# BATCH_SIZE = 128
# EMBED_SIZE = 128 # dimension of the word embedding vectors
# SKIP_WINDOW = 1 # the context window
# NUM_SAMPLED = 64    # Number of negative examples to sample.
# LEARNING_RATE = 1.0
# NUM_TRAIN_STEPS = 100000습
# WEIGHTS_FLD = 'processed/'
# SKIP_STEP = 2000
#
# ###Phase 1
# #define placeholders
# center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE]) #scalar 값이 들어갈 예정. 단어의 index를 이용해서 학습 (embed_matrix를 참조하게 됨)
# target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE]) #scalar 값이 들어갈 예정. 단어의 index를 이용해서 학습 (embed_matrix를 참조하게 됨)
#
# #define variable
# embed_matrix = tf.Variable(initial_value=tf.random_uniform(shape=[VOCAB_SIZE, EMBED_SIZE], minval=-1.0, maxval=1.0))
#
# #inference model
# #embedding matrix를 이용해서 lookup 형식으로 학습
# embed = tf.nn.embedding_lookup(params=embed_matrix, ids=center_words) #center_words에 해당하는 index(row)의 행 값. embed_matrix에서 찾는다.
#
# #define NCE loss
# nce_weight = tf.Variable(tf.truncated_normal(shape=[VOCAB_SIZE, EMBED_SIZE], stddev=1.0/EMBED_SIZE ** 0.5)) #이것 아니면 bias 부분의 shape가 이상한데...
# nce_bias = tf.Variable(tf.zeros(shape=[VOCAB_SIZE]))
# loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,        #학습할 weight
#                                      biases=nce_bias,            #학습할 bias
#                                      labels=target_words,       #target의 index
#                                      inputs=embed,              #update할 embedding lookup matrix
#                                      num_sampled=NUM_SAMPLED,   #sampling 진행할 개수
#                                      num_classes=VOCAB_SIZE))   #output 개수
#
# #define optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
#
#
#
# ###Phase 2
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter('./graphs', sess.graph)
#
#     average_loss = 0.0
#     for index in range(NUM_TRAIN_STEPS):
#         batch = next(batch_gen) #python 3에서는 batch_get.next()가 아니고 이렇게 표현해야 한다.
#         loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: batch[0], target_words: batch[1]})
#         average_loss += loss_batch
#         if (index+1) % 2000 == 0:
#             print('Average loss at step {}: {:5.1f}'.format(index+1, average_loss/(index+1)))
#
#     writer.close()
#
# #################################################################################################################










################################################################################################################
# 이제 전반적인 것들을 다 모아서 한번에 skip-gram 돌아가게 만들자.
################################################################################################################

#import modules
import os
import tensorflow as tf
from process_data import process_data
from tensorflow.contrib.tensorboard.plugins import projector

VOCAB_SIZE = 50000      #단어 50,000개
BATCH_SIZE = 128        #batch는 128개씩
EMBED_SIZE = 128        #임베딩 차원
SKIP_WINDOW = 1         #window의 크기
NUM_SAMPLED = 64        #negative sample 수
LEARNING_RATE = 1.0     #eta
NUM_TRAIN_STEPS = 20000 #2만 epoch
SKIP_STEP = 1000        #1000번마다 한번씩 loss를 출력해줌



class SkipGramModel:
    """ Build the graph for word2vec model """

    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate): #각종 parameter를 넣어서 Skip-gram 객체를 생성한다.
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    #data가 들어갈 placeholder를 생성한다.
    #문장에서 center(x)를 통해 target(y)를 예측하며, 이는 int32형으로 들어가서 embedding matrix의 row를 참조한다.
    #각각 batch 단위로 128개씩 들어가며, 특이한 점은 shape로, 형태가 input(x)는 row를 찾기위함이고, output(y)는 nce를 위한 형태이다.
    #tf.name_scope("data")를 통해 두개의 객체를 하나로 묶어서 나중에 tensorboard를 통해 확인할 때 시각적 편의를 제공한다.
    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    #embedding matrix를 만든다.
    #먼저 cpu 위에 생성하는데, 원한다면 gpu로 생성할 수 있다.
    #embed라는 이름으로 name_scope를 정의하고, 참조할 matrix의 크기는 vocab_size * embed_size로 결정한다.(input -> hidden층의 weight)
    #이를 uniform distribution(-1, 1)을 이용해서 초기화 해준다.
    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        with tf.device('/cpu:0'): #/gpu:0으로 지정해주면 된다.
            with tf.name_scope("embed"):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                                   self.embed_size], -1.0, 1.0),
                                                name='embed_matrix')

    #loss를 정의한다.
    #이 역시 cpu상에 생성하고, name_scope를 'loss'라 이름 붙인다.
    def _create_loss(self):
        """ Step 3 + 4: define the model + the loss function """
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                # Step 3: define the inference
                #위에서 정의한 embed_matrix에서 center_words에 해당하는 row의 vector를 가져와서 embed라고 한다.(길이는 embed_size)
                embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

                # Step 4: define loss function
                # construct variables for NCE loss
                #Word2Vec에서의 loss는 기본적인 softmax가 아닌 negative sampling, NCE등을 이용하는데 이 코드에서는, NCE를 사용한다.
                #nce를 위한 weight와 bias를 정의한다.
                #여기서 특이한 점이 embed_size*vocab_size여야 할 것 같은데 그 반대 순서로 크기가 정의되어 있다.(bias도 마찬가지)
                #이는 nce 논문을 읽어보고, tensor에서 구현한 코드를 좀 들여다 봐야 이해할 수 있을 듯 하다.
                nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                             stddev=1.0 / (self.embed_size ** 0.5)),
                                         name='nce_weight')
                nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

                # define loss function to be NCE loss function
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                          biases=nce_bias,
                                                          labels=self.target_words,
                                                          inputs=embed,
                                                          num_sampled=self.num_sampled,
                                                          num_classes=self.vocab_size), name='loss')

    #optimizer 정의
    #learning rate만큼 update를 gradient descent 기반으로 loss를 줄여나간다.
    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

    #학습 도중에 얻을 수 있는 loss 정보를 scalar형태나 histogram 형태로 모은다.
    #각각의 summary를 정의해둔 후 merge_all()을 통해 한번에 모아서 summary_op라는 이름의 필드에 저장한다.
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            # 여러개의 summary를 저장하고자 하므로 쉽게 관리하기 위해 전부를 merge한다.
            self.summary_op = tf.summary.merge_all()

    def build_graph(self): #위에서 정의한 5개의 메소드(placeholder, embedding, loss, optimizer, summaries를 실행하여 각각의 필드를 생성한다.
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


#모델을 학습하는 파트
#정의한 model과 데이터(batch_gen), 학습 epoch를 넣어준다.
def train_model(model, batch_gen, num_train_steps):

    #학습 전에 먼저 variables를 초기화 해준다.
    sess.run(tf.global_variables_initializer())

    #average loss 계산시 사용되는 변수
    total_loss = 0.0

    #tensorboard를 보기위해 writer 정의
    writer = tf.summary.FileWriter('./graphs/notes04/skip_gram_graph/lr' + str(LEARNING_RATE), sess.graph)

    #model의 global_step을 initial step으로 저장한다.
    initial_step = 0 #initial_step = global_step.eval()


    #처음 스텝부터 마지막까지 정의해서 학습한다.
    for index in range(initial_step, initial_step + num_train_steps):
        #batch_gen이라는 것으로부터 다음 배치를 받는다.
        centers, targets = next(batch_gen)

        #feed_dict를 위의 배치로 정의한다.
        feed_dict = {model.center_words: centers, model.target_words: targets}

        #run을 통해 loss, optimizer, summary_op를 각각의 이름으로 저장한다.
        loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
                                          feed_dict=feed_dict)

        #위의 line에서 summary를 add_summary라는 메소드를 이용해서 기록해준다.(tensorboard)
        writer.add_summary(summary, global_step=index)


        #total_loss에 계속 loss를 더해줌. (마지막에 전체 개수를 나눠줄 예정)
        total_loss += loss_batch

        #1000번 마다 한번씩 print 해줌
        if (index+1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index+1, total_loss / SKIP_STEP))
            total_loss = 0.0



# 저장용 폴더를 미리 만들어둬야 한다.
# os.makedirs('./graphs/notes04/skip_gram_checkpoints/')
# os.makedirs('./graphs/notes04/skip_gram_processed/')
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
train_model(model, batch_gen, NUM_TRAIN_STEPS)
# 모형 저장
saver.save(sess, './graphs/notes04/skip_gram_checkpoints/skipGram.ckpt')
sess.close()




################################################################################
# embedding 시각화
################################################################################
#skip gram model 객체 생성
model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
#model 내의 여러 정의할 부분을 하나의 메소드로 모두 실행
model.build_graph()
#학습한 parameter 불러오기(restore)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, './graphs/notes04/skip_gram_checkpoints/skipGram.ckpt')

# 위에서 학습한 embedding_matrix를 가져온다.
#(50000, 128)의 maxtix 이다.
final_embed_matrix = sess.run(model.embed_matrix)

# variable 형태만 작동할 수 있다. variable 타입으로 만들어주자.
# 이때, final_embed_matrix를 재사용하면 initialize를 학습한 내용으로 할 수 있다.
embedding_var = tf.Variable(initial_value=final_embed_matrix[:1000], name='embedding')
sess.run(embedding_var.initializer)

# 시각화 하기위한 객체
# summary를 통해서 저장해준다. (processed라는 폴더에 저장)
config = projector.ProjectorConfig()
summary_writer = tf.summary.FileWriter('./graphs/notes04/skip_gram_processed')

# embeddings의 add 메소드를 이용해서 embedding의 이름을 지정
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# word2vec을 통해 단어의 embedding 결과물을 보고자 하므로, 각 node에 맞는 word를 매칭하기 위한 metadata의 경로를 지정해준다.
embedding.metadata_path = './data/vocab_1000.tsv'

# config파일을 저장한다.
# 여기에는 embedding_var(단어 임베딩 정보)만을 담는다.
projector.visualize_embeddings(summary_writer, config)
saver_embed = tf.train.Saver([embedding_var])
saver_embed.save(sess, './graphs/notes04/skip_gram_processed/model_processed.ckpt')

# session 종료
sess.close()