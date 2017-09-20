#여러가지 변수로 인해 학습하고 있는 것들을 그때그때 저장해둘 필요가 있다.
#모형을 binary 형으로 저장해서 parameter를 실시간으로 관리해주자
#tensorflow에서는 Saver라는 클래스를 이용해서 저장하고자 하는 parameter를 저장 / 불러올 수 있다.

#import modules
import tensorflow as tf

#define model
# notes04에서 학습한 skip-gram 같은 model을 정의

#saver object 생성
saver = tf.train.Saver()


#매번 저장하는 파트
training_step = 10
with tf.Session() as sess:
    #actual trainig loop
    for step in range(training_step):
        sess.run([optimizer]) #optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss) 로 정의하여 학습
        if (step+1) % 1000 ==0:
            saver.save(sess, 'notes05_SaveAndRestore/checkpoint/skip-gram'+str(step)+'.ckpt') #이렇게 .ckpt로 확장자명을 알려주는것이 안전한 듯 하다.
            #혹은 아래와 같이 global step을 정의해뒀다면 이렇게도 save할 수 있음
            #중요한 것은 global_step 변수에서 trainable=False 옵션이다. 이렇게 해야 이 변수는 학습이 진행되지 않고, step만의 역할을 하게 된다.
            # global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')
            # saver.save(sess, 'notes05_SaveAndRestore/checkpoint/skip-gram',global_step=global_step)


#save 메소드를 이용하면 디폴트로 모든 variable들이 저장되며, 권고되는 부분이기도 하다. (graph전체가 저장되지 않음. 따라서 불러올때는 graph를 생성한 뒤 불러온다.)
#그런데, 특정 variable들만 저장하고 싶을때는 list, dict형태로 처음에 Saver()객체를 생성할때 할당해 주면 된다.
v1 = tf.Variable(name='v1')
v2 = tf.Variable(name='v2')
#ex1 : dict로 저장
saver = tf.train.Saver({'v1': v1, 'v2': v2})
#ex2 : list로 저장
saver = tf.train.Saver([v1, v2])
#ex3 : 이름을 key로, variable을 value로 저장
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})


##################################################################여기까지는 notes4에서 한 부분이므로 생략

# restore를 이용해서 해당 model의 학습된 parameter를 session으로 가져온다.
saver.restore(sess, 'skip-gram_checkpoints/skip-gram5.ckpt')

#아래 코드와 같이 미리 있는지 파악한 뒤에 가져오는 방식을 취할 수도 있다.
#directory에서 check point가 있는지 파악
ckpt = tf.train.get_checkpoint_state('skip-gram_checkpoints')
#가장 최근의 내용을 restore한다.
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)




