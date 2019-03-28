# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, \
  super, zip

import tensorflow as tf
import numpy as np
import random
import timeit
import collections
import nltk

TRAINING_SESSION = True

# 변수들 설정
rand = random.Random(0)
embed_size = 256 # 워드 임베딩 차원 크기
state_size = 256 # LSTM 히든 차원 크기
max_epochs = 100 # 최대 학습 횟수
minibatch_size = 20 # Minibatch 크기
min_token_freq = 3 # 단어 최소 등장 빈도

run_start = timeit.default_timer() # 시간 측정

print('Loading raw data...')
nltk.download('brown') # nltk에서 brown 말뭉치 다운로드

# all_seqs = [['and', 'how', 'right', 'she', 'was', '.'], ['the', 'operator', 'asked', 'pityingly', '.']]
# brown 말뭉치 문장(길이가 x 이상 y이하) 추출 -> 추출된 문장의 단어들을 소문자화 후 list에 보관
# all_seqs = [['i', 'am', 'a', 'boy],['you', 'are', 'a', 'girl']...]
all_seqs = [[token.lower() for token in seq] for seq in nltk.corpus.brown.sents() if 5 <= len(seq) <= 30]
rand.shuffle(all_seqs) # all_seqs 랜덤 섞기
all_seqs = all_seqs[:50000] # all_seqs 중에 20000개만 사용


trainingset_size = round(0.9 * len(all_seqs)) # 9할을 학습데이터로
train_seqs = all_seqs[:trainingset_size] # 처음부터 trainingset_size까지를 train 데이터로
validationset_size = len(all_seqs) - trainingset_size # 1할을 검증데이터로
val_seqs = all_seqs[-validationset_size:] # 처음부터 validationset_size까지를 valid 데이터로

print('Training set size:', trainingset_size)
print('Validation set size:', validationset_size)

all_tokens = (token for seq in train_seqs for token in seq) # all_tokens = ('and', 'how', 'right', ... )
token_freqs = collections.Counter(all_tokens) # {'boy': freq, 'girl':freq, ....}
vocab = sorted(token_freqs.keys(), key=lambda token: (-token_freqs[token], token)) # 단어 빈도수 정렬
# 단어 최소 등장 빈도 이하인 단어들 모두 제거
while token_freqs[vocab[-1]] < min_token_freq:
  vocab.pop()
vocab_size = len(vocab) + 2  # 문장 맨 앞, 맨 뒤 마커로 +1, 미등록어로 +1

token_to_index = {token: i + 2 for (i, token) in enumerate(vocab)} # 단어에 고유 번호 부여
index_to_token = {i + 2: token for (i, token) in enumerate(vocab)} # 고유 번호에 단어 부여
edge_index = 0 # 문장 맨 앞, 맨 뒤 위치
unknown_index = 1 # 미등록어 위치

print('Vocabulary size:', vocab_size)

# train, valid input을 생성하는 함수
def parse(seqs):
  indexes = list()
  lens = list()
  for seq in seqs:
    # 문장 속 단어들을 모두 고유 번호로 변환
    indexes_ = [token_to_index.get(token, unknown_index) for token in seq]
    indexes.append(indexes_) # 고유 번호로 표현된 문장들 list에 추가
    lens.append(len(indexes_) + 1)  # 모든 문장들의 길이 정보 저장 +1은 문장 맨 앞 혹은 맨 뒤 마커 추가 때문

  maxlen = max(lens) # 가장 긴 문장의 길이 정보

  in_mat = np.zeros((len(indexes), maxlen)) # LSTM 입력 문장 형태
  out_mat = np.zeros((len(indexes), maxlen)) # LSTM 출력 문장 형태
  for (row, indexes_) in enumerate(indexes):
    in_mat[row, :len(indexes_) + 1] = [edge_index] + indexes_ # 문장의 시작에 edge 마커 추가
    out_mat[row, :len(indexes_) + 1] = indexes_ + [edge_index] # 문장의 끝에 edge 마커 추가
  return (in_mat, out_mat, np.array(lens))

# train_seqs_in = [[0,423,23,8,656],[0,23,3,2,86]...]
# train_seqs_out = [[423,23,8,656,0],[23,3,2,86,0]...]
# train_seqs_len = [4,2,3,5, ...]
(train_seqs_in, train_seqs_out, train_seqs_len) = parse(train_seqs)
(val_seqs_in, val_seqs_out, val_seqs_len) = parse(val_seqs)

print('Training set max length:', train_seqs_in.shape[1] - 1)
print('Validation set max length:', val_seqs_in.shape[1] - 1)

################################################################
print()
print('Training...')

# Full correct sequence of token indexes with start token but without end token.
# LSTM의 입력
seq_in = tf.placeholder(tf.int32, shape=[None, None], name='seq_in')  # [seq, token]

# Length of sequences in seq_in.
seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')  # [seq]
tf.assert_equal(tf.shape(seq_in)[0], tf.shape(seq_len)[0])

# Full correct sequence of token indexes without start token but with end token.
# LSTM의 출력 즉 정답
seq_target = tf.placeholder(tf.int32, shape=[None, None], name='seq_target')  # [seq, token]
tf.assert_equal(tf.shape(seq_in), tf.shape(seq_target))

batch_size = tf.shape(seq_in)[0]  # Number of sequences to process at once.
num_steps = tf.shape(seq_in)[1]  # Number of tokens in generated sequence. # LSTM에서 순환의 횟수

# Mask of which positions in the matrix of sequences are actual labels as opposed to padding.
# 나중 학습 에러를 계산하기 위한 마스킹
# tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                  #  [True, True, True, False, False],
                                  #  [True, True, False, False, False]]
token_mask = tf.cast(tf.sequence_mask(seq_len, num_steps), tf.float32)  # [seq, token]

with tf.variable_scope('prefix_encoder'):
  # Encode each sequence prefix into a vector.

  # Embedding matrix for token vocabulary.
  # 임베딩 변수 선언 초기화 값은 xavier initializer 사용
  emb_shape = # embedding shape (TODO)
  embeddings = tf.get_variable('embeddings', emb_shape, tf.float32,
                               tf.contrib.layers.xavier_initializer())  # [vocabulary token, token feature]

  # 3D tensor of tokens in sequences replaced with their corresponding embedding vector.
  # seq_in [seq, token] -> [seq, token, emb_feature]로 lookup
  embedded = # tf.nn.embedding_lookup()  (TODO)

  # Use an LSTM to encode the generated prefix.
  # LSTM의 초기 state 선언 LSTM initial state = (c_vec,h_vec) 튜플로 구성됨
  init_state = # tf.contrib.rnn.LSTMStateTuple() (TODO)
  
  cell = tf.contrib.rnn.BasicLSTMCell(state_size) # 순환 신경망의 Cell을 정의
  # 실제 LSTM cell을 가지는 RNN 네트워크를 생성 return 값은 (output, state) 튜플
  prefix_vectors = # tf.nn.dynamic_rnn()[0]  (TODO)

with tf.variable_scope('softmax'):
  # Output a probability distribution over the token vocabulary (including the end token).
  # 최종 생성 단어 확률 분포를 구하는 부분 (state, state_size)*(state_size, vocab_size) + b = (state, vocab_size)
  W = tf.get_variable('W', [state_size, vocab_size], tf.float32, tf.contrib.layers.xavier_initializer())
  b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
  logits = tf.reshape(tf.matmul(tf.reshape(prefix_vectors, [-1, state_size]), W) + b,
                      [batch_size, num_steps, vocab_size])
  predictions = tf.nn.softmax(logits)  # [seq, prefix, token]

# 내부적으로 softmax를 취한 뒤 loss를 계산 한다. 따라서 입력이 logits이다. * token_mask로 해당사항 없는 것들은 0으로 만듬
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seq_target, logits=logits) * token_mask
total_loss = tf.reduce_sum(losses) # 문장 내 모든 단어에서 발생하는 loss의 합
train_step = tf.train.AdamOptimizer().minimize(total_loss)
saver = tf.train.Saver() # 변수들을 저장하고 불러오기 위한 클래스 초기화

sess = tf.Session()

if TRAINING_SESSION:
  # () 속의 텐서들을 계산하거나 또는 () 속의 오퍼레이션을 실행
  # 따라서 그래프 내의 변수들을 초기화하는 오퍼레이션을 실행
  sess.run(tf.global_variables_initializer())

  print('epoch', 'val loss', 'duration', sep='\t')

  epoch_start = timeit.default_timer()

  validation_loss = 0
  for i in range(len(val_seqs) // minibatch_size):
    # seq_in 0부터 minibatch_size 까지를 입력으로 i번째 minibatch를 수행
    # feed_dict는 그래프의 입력값, total_loss는 최종 그래프에서 구하게 될 텐서
    minibatch_validation_loss = sess.run(total_loss, feed_dict={
      seq_in: val_seqs_in[i * minibatch_size:(i + 1) * minibatch_size],
      seq_len: val_seqs_len[i * minibatch_size:(i + 1) * minibatch_size],
      seq_target: val_seqs_out[i * minibatch_size:(i + 1) * minibatch_size],
    })
    # 모든 minibatch들의 loss를 더해주는 것
    validation_loss += minibatch_validation_loss

  print(0, round(validation_loss, 3), round(timeit.default_timer() - epoch_start), sep='\t')
  last_validation_loss = validation_loss
  
  # 최초로 초기화 된 변수들을 모델로 저장함
  saver.save(sess, './model')

  #trainingset_indexes = [0,1,2,3,4,...len(train_seqs)]
  trainingset_indexes = list(range(len(train_seqs)))
  for epoch in range(1, max_epochs + 1):
    epoch_start = timeit.default_timer()

    rand.shuffle(trainingset_indexes)
    for i in range(len(trainingset_indexes) // minibatch_size):
      # validation하고 다르게 random suffle이 들어가서 아래 내용이 조금 달라진 것
      minibatch_indexes = trainingset_indexes[i * minibatch_size:(i + 1) * minibatch_size]
      a = sess.run([train_step, logits, seq_target], feed_dict={
        seq_in: train_seqs_in[minibatch_indexes],
        seq_len: train_seqs_len[minibatch_indexes],
        seq_target: train_seqs_out[minibatch_indexes],
      })

    validation_loss = 0
    for i in range(len(val_seqs) // minibatch_size):
      minibatch_validation_loss = sess.run(total_loss, feed_dict={
        seq_in: val_seqs_in[i * minibatch_size:(i + 1) * minibatch_size],
        seq_len: val_seqs_len[i * minibatch_size:(i + 1) * minibatch_size],
        seq_target: val_seqs_out[i * minibatch_size:(i + 1) * minibatch_size],
      })
      validation_loss += minibatch_validation_loss
      
    # early_stop 조건
    if validation_loss > last_validation_loss:
      break
    last_validation_loss = validation_loss

    saver.save(sess, './model')

    print(epoch, round(validation_loss, 3), round(timeit.default_timer() - epoch_start), sep='\t')

  print(epoch, round(validation_loss, 3), round(timeit.default_timer() - epoch_start), sep='\t')

################################################################
print()
print('Evaluating...')

# 저장된 모델 불러오는 것
saver.restore(sess, tf.train.latest_checkpoint('.'))

### 실습 함수
# def seq_prob(seq):
#   return ?

# print('P(the dog barked.) =', seq_prob(['the', 'dog', 'barked', '.']))
# print('P(the cat barked.) =', seq_prob(['the', 'cat', 'barked', '.']))
# print()