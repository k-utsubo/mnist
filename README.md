# mnist

* download data file

t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz

* extract data file

ex.
```gunzip t10k-images-idx3-ubyte.gz```

* convert data

ex.
``` conv.sh ```

* beginner model

regression by tensorflow
``` python tf_regression.py```

* multilayer perceptron model

```python tf_mlp.py```

* Convolutional Neural Network model

```python tf_cnn.py```
