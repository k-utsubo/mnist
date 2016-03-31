#!/bin/env python
# -*- coding: utf-8 -*-
# http://tensorflow.classcat.com/2016/02/11/tensorflow-how-tos-visualizing-learning/
import tensorflow as tf
import numpy as np
import random
import time

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

# 重み
# 訓練画像のpx数の行、ラベル（0-9の数字の個数）数の列の行列
# 初期値として0を入れておく
W = tf.Variable(tf.zeros([NUMPARAM, NUMCLASS]), name="weight")

# バイアス
# ラベル数の列の行列
# 初期値として0を入れておく
b = tf.Variable(tf.zeros([NUMCLASS]), name="bias")

# ソフトマックス回帰を実行
# yは入力x（画像）に対しそれがある数字である確率の分布
# matmul関数で行列xとWの掛け算を行った後、bを加算する。
# yは[1, 10]の行列
# name scope を使ってグラフ・ビジュアライザにおけるノードをまとめます。
with tf.name_scope("Wx_b") as scope:
  y = tf.nn.softmax(tf.matmul(x, W) + b)


# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()
# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver([W,b])
saver.restore(sess, "/tmp/tf_regression.ckpt")



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
