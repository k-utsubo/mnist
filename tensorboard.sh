#!/bin/sh

tensorboard --logdir=/tmp/data > /tmp/tensorboard.log 2>&1 &
