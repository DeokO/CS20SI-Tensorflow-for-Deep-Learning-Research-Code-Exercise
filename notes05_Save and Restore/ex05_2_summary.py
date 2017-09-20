#시각화 할때 matplotlib을 사용할 것이 아니라, tensorboard에서 제공하는 시각화 툴을 이용하는 것이 더 멋있다.
#보통 loss, average loss, accuracy등을 많이 시각화 한다.
#scalar plots, histograms, image로 시각화 하여 본다.


#아래의 예시 코드는 ex04_structModel.py 에서 있는 파트이다.(Skip-gram의 class 안에서 정의된 메소드)

import tensorflow as tf

def _create_summaryies(self):
    # summary name_scope를 정의한다.
    with tf.name_scope('summaries'):
        #loss를 scalar로 저장
        tf.summary.scalar('loss', self.loss)
        #loss를 histogram으로 저장
        tf.summary.histogram('hist_loss', self.loss)
        #accuracy를 scalar로 저장
        tf.summary.scalar('accuracy', self.accuracy)

        #각각의 summary 정보를 log로 남기고 이를 모두 모아서 한번에 session에서 내보내야 한다.
        #한번에 관리하기 쉽도록 하나의 op로 종합한다.
        self.summary_op = tf.summary.merge_all()
        #tf.summary.image(name, tensor, max_outputs=3, collections=None) 으로 image도 할 수 있다. 이것은 따로 추가 공부 필요

#위에서 정의된 summary_op는 op이므로, sess.run()을 통해 실행되어야 한다.
#model이라는 Skip-gram 객체에서 각각의 필드들을 sess.run을 통해 실행시켜준다.
#summary라는 이름으로 위에서 종합했던 summary_op를 실행시켜 저장했다.
loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)

#FileWriter 객체에 위에서 저장한 summary를 add_summary 메소드로 보내서 파일로 저장해준다.
writer = tf.summary.FileWriter('./skip-gram_graph')
writer.add_summary(summary, global_step=step)

#저장된 것은 ./skip-gram_graph 디렉토리에서 명령프롬프트를 이용해서 확인할 수 있다.
#tensorboard --logdir=./
#를 명령프롬프트에 적어주고 chrome에 localhost:6006을 url bar에 넣어주면 볼 수 있다.
