# 보통 데이터를 학습할 때 넣어주는 방식으로 feed_dict를 사용한다.
# 이보다 더 효율적으로 사용 가능한 방식이 있어서 강의에서 소개한다.
# 실제로는 feed_dict를 사용한 코드가 많은 것을 많이 볼 수 있다.
# 빠른 속도를 위해서는 모두 tensor단에서 돌리는게 좋으니 배워두면 좋을 듯 하다.

import tensorflow as tf


#본 데이터는 9개의 feature와 1개의 label이 있는 데이터이다. (# of row = 462)
#정수형도 있지만, float형으로만 고려하자.
#5번째 변수는 string형이다.
#label은 마지막 열이며 이는 int형이다. binary
DATA_PATH = './data/heart.csv'
BATCH_SIZE = 3
N_FEATURES = 9

def batch_generator(filenames):

    filename_queue = tf.train.string_input_producer(filenames) #filenames는 list 형태로 넣어주어 queue 형으로 받는다.
    #queue : 컴퓨터의 기본적인 자료구조의 한가지로, 먼저 집어넣은 데이터가 먼저 나오는(FIFO, First In First Out) 구조로 저장하는 형식을 말한다.
    reader = tf.TextLineReader(skip_header_lines=1) # 파일의 1번째 line은 header로 지정해준다.
    _, value = reader.read(filename_queue) #queue를 읽는다.

    #각 feature마다 missing값이 있다면, 어떤 값으로 채워줄지 결정한다.
    #대부분을 1.0(float)로 채우고, string변수는 ''로, label은 1로 missing value를 채운다.
    record_defaults = [[1.0] for _ in range(N_FEATURES)]
    record_defaults[4] = ['']
    record_defaults.append([1])

    # csv 파일을 불러오고, default 값까지도 전달해준다.
    content = tf.decode_csv(value, record_defaults=record_defaults)

    # 5번째 열이 string 형태인데(present/absent), 이것을 0과 1로 바꿔준다.
    condition = tf.equal(content[4], tf.constant('Present'))
    content[4] = tf.where(condition, tf.constant(1.0), tf.constant(0.0))

    # 9개의 변수를 모두 모아준다.
    features = tf.stack(content[:N_FEATURES])

    # 마지막 content는 label로 저장한다.
    label = content[-1]

    # minimum number elements in the queue after a dequeue
    min_after_dequeue = 10 * BATCH_SIZE

    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE,
                                        capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch

def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        #여러 threads를 이용하기 위한 queue runner를 사용하기 위해서는 Coordinator가 필요하다.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(10): # generate 10 batches
            features, labels = sess.run([data_batch, label_batch])
            print(features)
        coord.request_stop()
        coord.join(threads)

def main():
    data_batch, label_batch = batch_generator([DATA_PATH])
    generate_batches(data_batch, label_batch)

if __name__ == '__main__':
    main()