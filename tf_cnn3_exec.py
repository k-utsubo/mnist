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
  w_h1 = tf.Variable(tf.truncated_normal([NUMPARAM, 500],
                          stddev=1.0 / math.sqrt(float(NUMPARAM))),name='weights')
  b_h1 = tf.Variable(tf.zeros([500]),name='biases')

  h1 = tf.nn.sigmoid(tf.matmul(x, w_h1) + b_h1)
# hidden2
with tf.name_scope("hidden_layer2") as scope:
  w_h2 = tf.Variable(tf.truncated_normal([500, 300],
                          stddev=1.0 / math.sqrt(float(500))),name='weights')
  b_h2 = tf.Variable(tf.zeros([300]),name='biases')

  h2 = tf.nn.sigmoid(tf.matmul(h1, w_h2) + b_h2)
# softmax layer
with tf.name_scope("softmax_layer") as scope:
  w_o = tf.Variable(tf.truncated_normal([300, NUMCLASS],
                          stddev=1.0 / math.sqrt(float(300))),name='weights')
  b_o = tf.Variable(tf.zeros([NUMCLASS]),name='biases')

  y = tf.nn.softmax((tf.matmul(h2, w_o) + b_o))



# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()

# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）
sess = tf.Session()
sess.run(init)



saver = tf.train.Saver([w_h1,b_h1,w_h2,b_h2,w_o,b_o])
saver.restore(sess, "/tmp/tf_cnn.ckpt")


# 精度の実行と表示
# テストデータの画像とラベルで精度を確認する
# ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
test_label=label_data(test[:,0])
test_image=image_data(test)


# http://d.hatena.ne.jp/sugyan/20151124/1448292129
print "精度"
print(test_image[0])
res=(sess.run(y, feed_dict={x: [test_image[0]]}))
print res[0]


# 終了時刻
end_time = time.time()
print "終了時刻: " + str(end_time)
print "かかった時間: " + str(end_time - start_time)
