#!/usr/bin/env python
# coding: utf-8

# In[2]:

import tensorflow as tf

# In[4]:

@tf.function
def add(a, b):
    return a + b

@tf.function
def sub(a, b):
    return a - b

@tf.function
def mul(a, b):
    return a * b

@tf.function
def div(a, b):
    return a / b

# In[7]:

@tf.function
def matmul(a, b):
    return tf.matmul(a, b)

# In[8]:

@tf.function
def linear(m, x, c):
    return add(matmul(m, x), c)

# In[9]:

def f(x):
    if x > 0:
        x *= x
    return x


print(tf.autograph.to_code(f))
