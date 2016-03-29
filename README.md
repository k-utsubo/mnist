# mnist

1. download data file

t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz

1. extract data file

ex.
```gunzip t10k-images-idx3-ubyte.gz```

1. convert data

ex.
``` conv.sh ```

1. beginner model

regression by tensorflow
``` python tf_regression.py```

1. multilayer perceptron model

```python tf_mlp.py```

1. Convolutional Neural Network model

```python tf_cnn.py```
