###word2vec 구축 연습
'''
Phase 1: graph 구축
- placeholders, weights, inference model, loss, optimizer 정의하기

Phase 2: 학습
- model variables 초기화
- training data 이용해서 학습하기
- cost 계산 및 model parameter 적합

Phase 3: 시각화
- from tensorflow.contrib.tensorboard.plugins import projector를 이용
- tensorboard에서 embedding된 결과물을 시각적으로 확인
- metadata의 경로를 지정해주는 부분이 어려우니 잘 익혀야 함
'''



#########################################################################
# module import, hyper-parameter 설정
#########################################################################
import tensorflow as tf


#hyper-parameter
VOCAB_SIZE = 50000      #단어 50,000개
BATCH_SIZE = 128        #batch는 128개씩
EMBED_SIZE = 128        #임베딩 차원
SKIP_WINDOW = 1         #window의 크기
NUM_SAMPLED = 64        #negative sample 수
LEARNING_RATE = 1.0     #eta
NUM_TRAIN_STEPS = 20000 #2만 epoch
SKIP_STEP = 1000        #1000번마다 한번씩 loss를 출력해줌






#########################################################################
# class를 이용해 skip-gram 모형 생성
#########################################################################
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
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,                          #학습할 weight
                                                          biases=nce_bias,                             #학습할 bias
                                                          labels=self.target_words,                    #target의 index
                                                          inputs=embed,                                #update할 embedding lookup matrix
                                                          num_sampled=self.num_sampled,                #sampling 진행할 개수
                                                          num_classes=self.vocab_size), name='loss')   #output 개수

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




#########################################################################
# model 학습을 위한 함수 정의
#########################################################################
#정의한 model, 데이터(batch_gen), 학습 epoch를 넣어준다.
def train_model(sess, model, batch_gen, num_train_steps):

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
        #python 3에서는 batch_get.next()가 아니고 이렇게 표현해야 한다.
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


