#tensorflow로 모델링을 할 때, 많은 부분에서 randomize 하는 것을 발견할 수 있다.

import tensorflow as tf

#randomize를 하는 방식은 크게 두가지가 있다.
#1. operation level에서 seed 할당
#새로운 session이 진행될 때마다 randomize 재시작됨
#각각의 op단에서 random seed를 계속 가지고 있게 된다.
my_var = tf.Variable(tf.truncated_normal([3, 3], stddev=0.1, seed=0))
c = tf.random_uniform([], -10, 10, seed=2)
#하나의 sess에서 randomize를 두번 하게되면 seed=2인 상태에서 2번 c가 호출됨
#이 with를 몇번을 돌려도 같은 결과물이 나온다.
with tf.Session() as sess:
    print(sess.run(c)) # 3.57493
    print(sess.run(c)) # -5.97319

#각각의 sess에서 randomize를 각각 하게되면 seed=2인 상태에서 각각 처음의 난수를 가져오게 되어 같은 값이 나옴.
#op 단에서 각각 seed를 가지고 있는 형태
with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493
with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493

#이 역시도 같은 결과물이 나온다. 변수명만 다른 상황이다. (op 단에서 각각 seed를 가지고 있는 형태)
c = tf.random_uniform([], -10, 10, seed=2)
d = tf.random_uniform([], -10, 10, seed=2)
with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493
with tf.Session() as sess:
    print(sess.run(d)) # >> 3.57493




#2. graph level에서 seed 할당
#여러 사람이 같은 코드를 돌리는데, 똑같은 결과가 나오고자 한다면 graph level에서 seed를 주고 진행하면 코드가 제대로 돌아가는지 확인할 수 있다.
#강의같은데서 사용 가능할 것이다.
seed=2
tf.set_random_seed(seed)


#원래대로라면 이렇게 각각 다른 값이 나오게 된다.
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)
#돌릴 때마다 다른 결과물이 나온다.
with tf.Session() as sess:
    print(sess.run(c)) # 4.24999
    print(sess.run(d)) # 0.664539

#만약 graph단에서 같은 seed를 주게된다면 두개의 서로다른 파일이 같은 결과를 출력하는 것을 확인할 수 있다.
#계속 with문만 돌리면 다른 결과물이 나오는데, 아예 새로 console을 켜서 seed를 설정하고 아래를 돌리면 계속 똑같이 -4.00, -2.98이 나오는 것을 확인할 수 있다.
seed=2
tf.set_random_seed(seed)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)
with tf.Session() as sess:
    print(sess.run(c)) # -4.00752
    print(sess.run(d)) # -2.98339


