#!/usr/bin/env python
# coding: utf-8

# In[55]:

import os, datetime

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# In[56]:

# Life Expectancy Dataset https://www.kaggle.com/kumarajarshi/life-expectancy-who
data = pd.read_csv('./datasets/Life Expectancy Data.csv')
data.sample(5)

# In[57]:

data.shape

# In[58]:

data.isna().sum()

# In[59]:

countries = data['Country'].unique()

na_cols = ['Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ', 'Polio', 'Total expenditure',
           'Diphtheria ', 'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Population',
           'Income composition of resources']

for col in na_cols:
    for country in countries:
        data.loc[data['Country'] == country, col] = data.loc[data['Country'] == country, col]            .fillna(data[data['Country'] == country][col].mean())

# In[60]:

data.isna().sum()

# In[61]:

data = data.dropna()
data.shape

# In[62]:

data['Status'].value_counts()

# In[63]:

data['Country'].value_counts()

# In[64]:

plt.figure(figsize=(10,8))

data.boxplot('Life expectancy ')

plt.show()

# In[65]:

plt.figure(figsize=(8,6))

sns.boxplot(x = data['Status'], y = data['Life expectancy '])

plt.xlabel('Status', fontsize = 16)
plt.ylabel('Total expenditure', fontsize = 16)

plt.show()

# In[66]:

plt.figure(figsize=(8,6))

sns.boxplot(x = data['Status'], y = data['Total expenditure'])

plt.xlabel('Status', fontsize = 16)
plt.ylabel('Total expenditure', fontsize = 16)

plt.show()

# In[67]:

data_corr = data[['Life expectancy ', 'Adult Mortality', 'Schooling', 'Total expenditure',
                  'Diphtheria ', 'GDP', 'Population']].corr()
data_corr

# In[68]:

fix, ax = plt.subplots(figsize=(12,8))

sns.heatmap(data_corr, annot=True)

plt.show()

# In[69]:

features = data.drop('Life expectancy ', axis=1)
target = data[['Life expectancy ']]

# In[70]:

features.columns

# In[71]:

target.sample(5)

# In[72]:

features = features.drop('Country', axis=1)
features.columns

# In[19]:

categorical_features = features['Status'].copy()
categorical_features.head()

# In[20]:

categorical_features = pd.get_dummies(categorical_features)
categorical_features.head()

# In[21]:

numeric_features = features.drop(['Status'], axis=1)
numeric_features.head()

# In[22]:

numeric_features.describe().T

# In[23]:

# NeuralModel tends to be more robust. They are trained using numeric features with same scale.
# We achieve this using Standardization
standardScaler = StandardScaler()

numeric_features = pd.DataFrame(standardScaler.fit_transform(numeric_features),
                                columns=numeric_features.columns,
                                index=numeric_features.index)
numeric_features.describe().T

# In[24]:

processed_features = pd.concat([numeric_features, categorical_features], axis=1, sort=False)
processed_features.head()

# In[25]:

processed_features.shape

# In[26]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(processed_features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=1)
(x_train.shape, x_test.shape), (y_train.shape, y_test.shape)

# In[27]:

def build_single_layer_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(32,
                                    input_shape = (x_train.shape[1],),
                                    activation = 'sigmoid'))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # mse: Mean Square Error
    # mae: Mean Absolute Error
    model.compile(loss = 'mse',
                  metrics = ['mae', 'mse'],
                  optimizer = optimizer)
    return model


# In[28]:

model = build_single_layer_model()
model.summary()

# In[29]:

tf.keras.utils.plot_model(model)

# In[30]:

num_epochs = 100

training_history = model.fit(x_train, y_train,
                             epochs = num_epochs,
                             validation_split = 0.2,
                             verbose = True)

# In[31]:

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)

plt.plot(training_history.history['mae'])
plt.plot(training_history.history['val_mae'])

plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'val'])

plt.subplot(1,2,2)

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])

# In[32]:

model.evaluate(x_test,y_test)

# In[33]:

# R square Score to better evaluate the model. Computes how much variants of the underline data have been captured.
# The highest the value the better
y_pred = model.predict(x_test)
r2_score(y_test, y_pred)

# In[34]:

pred_results = pd.DataFrame({'y_test': y_test.values.flatten(),
                             'y_pred': y_pred.flatten()}, index = range(len(y_pred)))
pred_results.sample(10)

# In[35]:

plt.figure(figsize=(10,8))

plt.scatter(y_test, y_pred, s=100, c='blue')

plt.ylabel('Actual life expectancy values')
plt.xlabel('Predicted life expectancy values')
plt.show()

# In[36]:

# Multiple layer model layers.Dense(number of neurons, ....)
def build_multiple_layer_model():
    model = tf.keras.Sequential([layers.Dense(32, input_shape = (x_train.shape[1],), activation = 'relu'),
                                 layers.Dense(16, activation = 'relu'),
                                 layers.Dense(4, activation = 'relu'),
                                 layers.Dense(1)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # mse: Mean Square Error
    # mae: Mean Absolute Error
    model.compile(loss = 'mse',
                  metrics = ['mae', 'mse'],
                  optimizer = optimizer)
    return model


# In[37]:

model = build_multiple_layer_model()
tf.keras.utils.plot_model(model, show_shapes=True)

# In[38]:

# remove logs
get_ipython().system('rm -r seq_logs')
get_ipython().system('ls -l')

# In[39]:

# Callbacks in TensorFlow : Customize the behavior of the model during training, evaluation, predictions
logdir = os.path.join("seq_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# In[ ]:



# In[40]:

training_history = model.fit(x_train, y_train,
                             validation_split = 0.2,
                             epochs = 500,
                             batch_size = 100,
                             callbacks = [tensorboard_callback])

# In[41]:

get_ipython().run_line_magic('load_ext', 'tensorboard')

# In[42]:

get_ipython().run_line_magic('tensorboard', '--logdir seq_logs --port 6050')

# In[43]:

model.evaluate(x_test, y_test)

# In[44]:

y_pred = model.predict(x_test)
r2_score(y_test, y_pred)

# In[45]:

# Yet another model
def build_model_with_sgd():
    model = tf.keras.Sequential([layers.Dense(32, input_shape = (x_train.shape[1],), activation = 'relu'),
                                 layers.Dense(16, activation = 'relu'),
                                 layers.Dense(4, activation = 'relu'),
                                 layers.Dense(1)])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    model.compile(loss = 'mse',
                  metrics = ['mae', 'mse'],
                  optimizer = optimizer)
    return model


# In[46]:

model_sgd = build_model_with_sgd()

tf.keras.utils.plot_model(model_sgd, show_shapes=True)

# In[47]:

training_history = model_sgd.fit(x_train, y_train,
                                 validation_split = 0.2,
                                 epochs = 100,
                                 batch_size = 100)

# In[48]:

model_sgd.evaluate(x_test, y_test)

# In[49]:

y_pred = model_sgd.predict(x_test)
r2_score(y_test, y_pred)

# In[50]:

# elu: exponential linear unit
def build_model_with_rmsprop():
    model = tf.keras.Sequential([layers.Dense(16, input_shape = (x_train.shape[1],), activation = 'elu'),
                                 layers.Dense(8, activation = 'elu'),
                                 layers.Dense(4, activation = 'elu'),
                                 layers.Dense(1)])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(loss = 'mse',
                  metrics = ['mae', 'mse'],
                  optimizer = optimizer)
    return model


# In[51]:

model_rms = build_model_with_rmsprop()

# In[52]:

training_history = model_rms.fit(x_train, y_train,
                                 validation_split = 0.2,
                                 epochs = 100,
                                 batch_size = 100)

# In[53]:

model_rms.evaluate(x_test, y_test)

# In[54]:

y_pred = model_rms.predict(x_test)
r2_score(y_test, y_pred)
