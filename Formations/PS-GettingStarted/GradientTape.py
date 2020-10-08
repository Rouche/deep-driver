#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# In[ ]:

x = tf.Variable(4.0)

with tf.GradientTape() as tape:
    y = x**2

# In[ ]:

y

# In[ ]:

dy_dx = tape.gradient(y, x)
dy_dx

# In[ ]:

w = tf.Variable(tf.random.normal((4,2)))
w

# In[ ]:

b = tf.Variable(tf.ones(2, dtype=tf.float32))
b

# In[ ]:

x = tf.Variable([[10., 20., 30., 40.]], dtype=tf.float32)

# In[ ]:

with tf.GradientTape(persistent=True) as tape2:
    y = tf.matmul(x, w) + b

    loss = tf.reduce_mean(y**2)

# In[ ]:

[dl_dw, dl_db] = tape2.gradient(loss, [w,b])

# In[ ]:

dl_dw

# In[ ]:

dl_db

# In[ ]:

layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[10., 20., 30.]])

# In[ ]:

with tf.GradientTape() as tape3:
    y = layer(x)

    loss = tf.reduce_sum(y**2)

grad = tape3.gradient(loss, layer.trainable_variables)

# In[ ]:

grad

# In[ ]:

# GradientTape watches trainable variables. Tensors, Constants, non-trainable variables are not tracked automatically.

# In[ ]:

# trainable by default
x1 = tf.Variable(5.0)
x1

# In[ ]:

x2 = tf.Variable(5.0, trainable=False)
x3 = tf.add(x1,x2)
x3

# In[ ]:

x4 = tf.constant(5.0)
x4

# In[ ]:

with tf.GradientTape() as tape4:
    y = (x1**2) + (x2**2) + (x3**2) + (x4**2)

grad = tape4.gradient(y, [x1, x2, x3, x4])
grad

# In[ ]:

# Make them trainable
x1 = tf.constant(5.0)
x2 = tf.Variable(3.0)

# In[ ]:

with tf.GradientTape() as tape5:
    tape5.watch(x1)

    y = (x1**2) + (x2**2)

# In[ ]:

[dy_dx1, dy_dx2] = tape5.gradient(y, [x1,x2])
dy_dx1, dy_dx2

# In[2]:

# Turn off statefull

x1 = tf.constant(5.0)
x2 = tf.Variable(3.0)

# In[3]:

with tf.GradientTape(watch_accessed_variables=False) as tape6:
    tape6.watch(x1)
    y = (x1**2) + (x2**2)

# In[4]:

[dy_dx1, dy_dx2] = tape6.gradient(y, [x1,x2])
dy_dx1, dy_dx2

# In[5]:

# GradientTape tracks actual operations, Control flow statements, branches are handled automatically
x = tf.constant(1.0)
x1 = tf.Variable(5.0)
x2 = tf.Variable(3.0)

# In[7]:

with tf.GradientTape(persistent=True) as tape7:
    tape7.watch(x)

    if x > 0.0:
        result = x1**2
    else:
        result = x2**2

dx1, dx2 = tape7.gradient(result, [x1, x2])

dx1, dx2

# In[8]:

x = tf.constant(-1.0)
x1 = tf.Variable(5.0)
x2 = tf.Variable(3.0)

# In[9]:

with tf.GradientTape(persistent=True) as tape7:
    tape7.watch(x)

    if x > 0.0:
        result = x1**2
    else:
        result = x2**2

dx1, dx2 = tape7.gradient(result, [x1, x2])

dx1, dx2
