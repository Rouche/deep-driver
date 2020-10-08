#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
tf.executing_eagerly()

# In[3]:

x = [[10.0]]
res  =tf.matmul(x,x)
res

# In[4]:

a = tf.constant([[10,20],[30,40]])
a

# In[5]:

b = tf.add(a, 2)
print(b)

# In[6]:

print(a*b)

# In[7]:

m = tf.Variable([4, 5, 6])
c = tf.Variable([1, 1, 1])

# In[12]:

x = tf.Variable([10,10,10])

# In[13]:

y = m*x+c

# In[14]:

y