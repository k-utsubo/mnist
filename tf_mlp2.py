#!/bin/env python
# -*- coding: utf-8 -*-
# http://qiita.com/ikki8412/items/95bc81a744dc377d9119
import tensorflow as tf
import numpy as np
import random
import time
import math

NUMCLASS=10
NUMPARAM=784
NUMHIDDEN=625
   
### データ処理用
def label_data(lines):
  labels=[]
  for line in lines:
    # ラベルを1-of-k方式で用意する
    tmp = np.zeros(NUMCLASS)
    tmp[int(line)] = 1
    labels.append(tmp)
  return np.asarray(labels)

def image_data(test):
  test_image=map(lambda n: map(lambda k: float(k)/255.0,n),test[:,1:NUMPARAM+1])
  return np.asarray(test_image)


# 開始時刻
start_time = time.time()
print "開始時刻: " + str(start_time)


### データ取得 --->
# ファイルを開く
f = open("train.txt", 'r')
# データを入れる配列
train = []
for line in f:
    # 改行を除いてスペース区切りにする
    line = line.rstrip()
    l = line.split(" ")
    l = map(lambda n: int(n),l)
    #l=map(lambda n: 0 if n=="0" else 1,l)
    train.append(l)


# numpy形式に変換
train = np.asarray(train)
f.close()

f = open("t10k.txt", 'r')
test = []
for line in f:
    line = line.rstrip()
    l = line.split(" ")
    l = map(lambda n: int(n),l)
    #l=map(lambda n: 0 if n=="0" else 1,l)
    test.append(l)

test = np.asarray(test)
f.close()
### データ取得 ---<


# 訓練画像を入れる変数
# 訓練画像は28x28pxであり、これらを1行784列のベクトルに並び替え格納する
# Noneとなっているのは訓練画像がいくつでも入れられるようにするため
x = tf.placeholder(tf.float32, [None, NUMPARAM], name="x-input")

# 交差エントロピー
# y_は正解データのラベル
# 損失とオプティマイザを定義します
y_ = tf.placeholder(tf.float32, [None, NUMCLASS], name="y-input")


# hidden1
with tf.name_scope("hidden_layer1") as scope:
  w_h = tf.Variable(tf.random_normal([NUMPARAM, NUMHIDDEN],mean=0.0, stddev=0.05))
  b_h = tf.Variable(tf.zeros([NUMHIDDEN]),name='biases')

  h = tf.sigmoid(tf.matmul(x, w_h) + b_h)
# output layer
with tf.name_scope("output_layer") as scope:
  w_o = tf.Variable(tf.truncated_normal([NUMHIDDEN, NUMCLASS],mean=0.0, stddev=0.05))
  b_o = tf.Variable(tf.zeros([NUMCLASS]),name='biases')

  y = tf.nn.softmax((tf.matmul(h, w_o) + b_o))


# 更なる name scopes はグラフ表現をクリーンアップしま
with tf.name_scope("xent") as scope:
  # Cost Function basic term
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  
  # Regularization terms (weight decay)
  L2_sqr = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
  lambda_2 = 0.01
  # the loss and accuracy
  loss = cross_entropy + lambda_2 * L2_sqr

  # TensorBoardで表示するよう指定
  tf.scalar_summary("cross_entropy", cross_entropy)

  # 勾配硬化法を用い交差エントロピーが最小となるようyを最適化する
  train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()

# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）
sess = tf.Session()
sess.run(init)
# TensorBoardで表示する値の設定
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter("/tmp/data", sess.graph_def)


# 1000回の訓練（train_step）を実行する
# next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
# feed_dictでplaceholderに値を入力することができる
print "--- 訓練開始 ---"
for i in range(20000):
  train_sample=np.asarray(random.sample(train,100))
  batch_ys=label_data(train_sample[:,0])
  batch_xs=image_data(train_sample)
  train_accuracy=sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

  # 1 step終わるたびにTensorBoardに表示する値を追加する
  summary_str=sess.run(summary_op, feed_dict={x: batch_xs, y_:batch_ys})
  summary_writer.add_summary(summary_str, i)
print "--- 訓練終了 ---"

# 正しいかの予測
# 計算された画像がどの数字であるかの予測yと正解ラベルy_を比較する
# 同じ値であればTrueが返される
# argmaxは配列の中で一番値の大きい箇所のindexが返される
# 一番値が大きいindexということは、それがその数字である確率が一番大きいということ
# Trueが返ってくるということは訓練した結果と回答が同じということ
with tf.name_scope("test") as scope:
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 精度の計算
# correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
# Trueならば1、Falseならば0に変換される
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  tf.scalar_summary("accuracy", accuracy)

# 精度の実行と表示
# テストデータの画像とラベルで精度を確認する
# ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
test_label=label_data(test[:,0])
test_image=image_data(test)
print "精度"
print(sess.run(accuracy, feed_dict={x: test_image, y_: test_label}))

# 終了時刻
end_time = time.time()
print "終了時刻: " + str(end_time)
print "かかった時間: " + str(end_time - start_time)
