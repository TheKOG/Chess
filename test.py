#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
from keras.utils.np_utils import to_categorical
import numpy as np
from numpy.core.defchararray import array
from numpy.core.fromnumeric import cumprod, shape
from numpy.core.numeric import identity
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()


a=np.array([[1,2],[3,4]])
a_=a.reshape(4)
b=7
c=np.hstack((a_,[b]))
print(c)
d=c[:4]
d=d.reshape((2,2))
print(d)