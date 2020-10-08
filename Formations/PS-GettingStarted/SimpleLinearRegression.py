#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# In[2]:

W_true = 2
b_true = 0.5

# In[3]:

x = np.linspace(0, 3, 130)
y = W_true * x + b_true + np.random.randn(*x.shape) * 0.5

# In[4]:

plt.figure(figsize=(10, 8))
plt.scatter(x, y)

plt.xlabel('x')
plt.ylabel('y')

plt.title("Training data")
plt.show()

# In[5]:

class LinearModel:

    def __init__(self):
        self.weight = tf.Variable(np.random.randn(), name="W")
        self.bias = tf.Variable(np.random.randn(), name="b")

    def __call__(self, x):
        return self.weight * x + self.bias

# In[6]:

def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

# In[7]:

def train(linear_model, x, y, lr=0.01):
    with tf.GradientTape() as tape:
        y_pred = linear_model(x)
        current_loss = loss(y, y_pred)

    d_weight, d_bias = tape.gradient(current_loss,
                                     [linear_model.weight, linear_model.bias])

    linear_model.weight.assign_sub(lr * d_weight)
    linear_model.bias.assign_sub(lr * d_bias)

# In[8]:

linear_model = LinearModel()

weights, biases = [], []

# Original 10 but better training 50
epochs = 100

lr = 0.15

# In[9]:

for epoch_count in range(epochs):
    weights.append(linear_model.weight.numpy())
    biases.append(linear_model.bias.numpy())

    real_loss = loss(y, linear_model(x))

    train(linear_model, x, y, lr=lr)

    print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")

# In[10]:

plt.figure(figsize=(10, 8))
plt.plot(range(epochs), weights, 'r', range(epochs), biases, 'b')
plt.plot([W_true] * epochs, 'r--', [b_true] * epochs, 'b--')

plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()

# In[11]:

linear_model.weight.numpy(), linear_model.bias.numpy()

# In[12]:

rmse = loss(y, linear_model(x))
rmse.numpy()

# In[13]:

# original dataset
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, linear_model(x), label='Fitted line')

plt.title('Linear regression')

plt.legend()
plt.show()

# In[14]:

from tensorflow import keras
from tensorflow.keras import layers

# In[15]:

x.shape, y.shape

# In[16]:

x = pd.DataFrame(x, columns=['x'])
y = pd.DataFrame(y, columns=['y'])

# In[17]:

x.head()

# In[18]:

y.head()

# In[19]:

x.shape, y.shape

# In[20]:

# Layer will feed the layer after
model = keras.Sequential([layers.Dense(1, input_shape=(1,), activation='linear')])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)

# In[21]:

model.fit(x, y, epochs=100)

# In[22]:

y_pred = model.predict(x)

# In[23]:

plt.figure(figsize=(10, 8))
plt.scatter(x, y, c='blue', label='Original data')
plt.plot(x, y_pred, color='r', label='Fitted line')

plt.title('Linear regression')
plt.legend()
plt.show()

# Keras Building blocks
# Sequential model : simple stack of layers, cannot be used for complex model. tf.keras.Sequential
# Functional APIs
# Model subclassing
# Custom layers
