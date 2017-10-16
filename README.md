# CS20SI-Tensorflow-for-Deep-Learning-Research
CS20SI: Tensorflow for Deep Learning Research url: https://web.stanford.edu/class/cs20si/syllabus.html

-----------------------------------------------------------

CS 20SI: Tensorflow for Deep Learning Research 강의를 공부하면서 정리한 내용입니다.

## notes01: Overview of Tensorflow
1. notes_01.pdf
2. slides_01.pdf

## notes02: Operations
1. notes_02.pdf
2. slides_02.pdf
3. ex02_operations.py
    - tensorflow 객체 생성, 기본 method 사용, graph 저장 등

## notes03: Regression
1. notes_03.pdf
2. slides_03.pdf
3. ex03_1_regression.py
    - 기본 회귀 모형 적합. 1차 함수 fitting, 2차 함수 fitting, plotting, optimizer 종류
4. ex03_2_logisticRegression.py
    - 기본 분류 모형 적합. softmax_cross_entropy_with_logits 이용한 학습

## notes04: StructModel
1. notes_04.pdf
2. slides_04.pdf
3. ex04_1_structModel.py
    - word2vec(skip-gram) 모형 구축 class 정의, 학습 function 정의
4. ex04_2_trainModel.py
    - 모형 학습 및 파라미터 저장(graphs/notes04/skip_gram_checkpoints에 저장)
5. ex04_3_visualization.py
    - Tensorboard에 임베딩 결과물 시각화(graphs/notes04/skip_gram_visualization 에 저장)

## notes05: Save and Restore
1. notes_05.pdf
2. slides_05.pdf
3. ex05_1_saveAndRestore.py
    - 학습한 모형의 parameter를 저장하는 방법과 읽어오는 방법
4. ex05_2_summary.py
    - tensorboard에 학습 결과를 시각화 하기 위한 summary 사용법
5. ex05_3_randomization.py
    - variable 정의할 때 많이 사용되는 random에 대한 공부
6. ex05_6_csvReader.py
    - csv 파일을 feed_dict 형태가 아닌 tensor에서 queue로 읽는 방법

## notes06: CNN and Neural style
1. slides_06

## notes07: Convolutional Neural Network
1. notes_07.pdf
2. slides_07.pdf
3. ex07_convNet.py
    - conv2d에 대한 설명
    - convNet 구조 생성 및 학습, summary, parameter 저장까지 모두 포함돼 있는 코드
    - input, output의 size를 명시해두어 이해하기에 편리
    - tf.variable_scope를 이용해 tensorboard에서 graph를 가독성 좋게 보도록 구성
    - 강의에서 제공하는 코드에서 많은 부분 추가됨

## notes09: Pipeline
1. notes_09.pdf
2. slides_09.pdf
3. ex09_Reader.py
    - ex05_4_csvReader.py 파일과 동일한 파일
    - 이 코드를 기준으로 csv에 대한 Reader를 실습
    - slide에는 Reader내용에 추가로 style transfer의 loss, optimization와 관련한 내용이 있음

## notes11: RNN
1. slides_11.pdf
2. extra01_MNIST_GRUCell.py
    - cs20si 11장에서는 chatbot 설명함
    - chatbot에 들어가기 앞서 기본 RNN 모형을 공부
    - golbin님의 코드를 참고하여 아래 내용을 추가
        - vanilla RNN, LSTM, GRU cell 모두를 적용 가능하게 접근
        - RNN cell에 대해 dropout 적용(tf.contrib.rnn.DropoutWrapper 이용)
        - Fully connected 파트는 tf.contrib.layers를 이용하였으며, batch normalization 적용
