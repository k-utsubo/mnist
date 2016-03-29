#!/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import random
import time

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

NUMCLASS=10
NUMPARAM=784
HIDDEN_UNIT_SIZE= 32

# 開始時刻
start_time = time.time()
print "開始時刻: " + str(start_time)


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


# Variables
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

w_h = tf.Variable(tf.random_normal([784, 625], mean=0.0, stddev=0.05))
w_o = tf.Variable(tf.random_normal([625, 10], mean=0.0, stddev=0.05))
b_h = tf.Variable(tf.zeros([625]))
b_o = tf.Variable(tf.zeros([10]))

# Create the model
def model(X, w_h, b_h, w_o, b_o):
    h = tf.sigmoid(tf.matmul(X, w_h) + b_h)
    pyx = tf.nn.softmax(tf.matmul(h, w_o) + b_o)

    return pyx

y_hypo = model(x, w_h, b_h, w_o, b_o)

# Cost Function basic term
cross_entropy = -tf.reduce_sum(y_*tf.log(y_hypo))

# Regularization terms (weight decay)
L2_sqr = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
lambda_2 = 0.01

# the loss and accuracy
loss = cross_entropy + lambda_2 * L2_sqr
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_hypo,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



#### training
init = tf.initialize_all_variables()

print "--- 訓練開始 ---"
with tf.Session() as sess:
    sess.run(init)
    for i in range(20001):
        train_sample=np.asarray(random.sample(train,100))
        batch_ys=label_data(train_sample[:,0])
        batch_xs=image_data(train_sample)
        train_step.run({x: batch_xs, y_: batch_ys})
        if i % 2000 == 0:
            train_accuracy = accuracy.eval({x: batch_xs, y_: batch_ys})
            print('  step, accurary = %6d: %6.3f' % (i, train_accuracy))

    # Test trained model
    test_label=label_data(test[:,0])
    test_image=image_data(test)
    print('accuracy = ', accuracy.eval({x: test_image, y_: test_label}))


# 終了時刻
end_time = time.time()
print "終了時刻: " + str(end_time)
print "かかった時間: " + str(end_time - start_time)
