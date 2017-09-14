#########################################################################
# module import, hyper-parameter 설정
#########################################################################
import tensorflow as tf
from notes04_StructModel.ex04_1_structModel import *
from tensorflow.contrib.tensorboard.plugins import projector



#########################################################################
# embedding 시각화
#########################################################################
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
summary_writer = tf.summary.FileWriter('./graphs/notes04/skip_gram_visualization')

# embeddings의 add 메소드를 이용해서 embedding의 이름을 지정
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# word2vec을 통해 단어의 embedding 결과물을 보고자 하므로, 각 node에 맞는 word를 매칭하기 위한 metadata의 경로를 지정해준다.
# 이 부분이 중요한데, 경로 지정을 단순하게 열린 logdir/metadata.tsv형태로 해주어야 한다.
# 만약에 경로를 지정해주고자 './graphs/notes04/skip_gram_visualization/vocab_1000.tsv'로 해주게 되면
# tensorboard에서 metadata의 path를 './graphs/notes04/skip_gram_visualization/graphs/notes04/skip_gram_visualization/vocab_1000.tsv'
# 이런식으로 중복해서 받게 되어 parsing metadata... 라는 오류 문구와 함께 작동하지 않는 것을 확인할 수 있다.
# 따라서 아래와 같이 그냥 logdir에 있는 file명으로 지정해주면 된다.
embedding.metadata_path = './vocab_1000.tsv'

# config파일을 저장한다.
# 여기에는 embedding_var(단어 임베딩 정보)만을 담는다.
projector.visualize_embeddings(summary_writer, config)
saver_embed = tf.train.Saver([embedding_var])
saver_embed.save(sess, './graphs/notes04/skip_gram_visualization/model_processed.ckpt')

# session 종료
sess.close()