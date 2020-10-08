#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# In[ ]:

wine_data = datasets.load_wine()

# In[ ]:

print(wine_data['DESCR'])

# In[ ]:

data = pd.DataFrame(data=wine_data['data'], columns=wine_data['feature_names'])

data['target'] = wine_data['target']

data.sample(5)

# In[ ]:

data.shape

# In[ ]:

data.describe().T

# In[ ]:

# Count the number of null values
data.isna().sum()

# In[ ]:

data['target'].value_counts()

# In[ ]:

sns.displot(data['alcohol'], kde=1)

# In[ ]:

plt.figure(figsize=(10, 6))

sns.boxplot(x=data['target'], y=data['alcohol'])

plt.xlabel('class', fontsize=20)
plt.xlabel('alcohol', fontsize=20)

plt.show()

# In[ ]:

features = data.drop('target', axis=1)

target = data[['target']]

# In[ ]:

features.columns

# In[ ]:

target.sample(5)

# In[ ]:

target = to_categorical(target, 3)
target

# In[ ]:

standardScaler = StandardScaler()

processed_features = pd.DataFrame(standardScaler.fit_transform(features),
                                  columns=features.columns,
                                  index=features.index)
processed_features.describe().T

# In[ ]:

x_train, x_test, y_train, y_test = train_test_split(processed_features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=1)

# In[ ]:

x_train.shape, y_train.shape

# In[ ]:

x_test.shape, y_test.shape

# In[ ]:

# Extends tf.keras.Model
class WineClassificationModel(Model):

    def __init__(self, input_shape):
        super(WineClassificationModel, self).__init__()

        self.d1 = layers.Dense(128, activation='relu', input_shape=[input_shape])
        self.d2 = layers.Dense(64, activation='relu')

        self.d3 = layers.Dense(3, activation='softmax')

    def call(self, x):
        x.self.d1(x)
        x.self.d2(x)

        x.self.d3(x)

        return x

# In[ ]:

model = WineClassificationModel(x_train.shape[1])

model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# In[ ]:

num_epochs = 500

# In[ ]:

training_history = model.fit(x_train.values, y_train,
                             validation_split=0.2,
                             epochs=num_epochs,
                             batch_size=48)

# In[ ]:

training_history.history.keys()

# In[ ]:

train_acc = training_history.history['accuracy']
train_loss = training_history.history['loss']

precision = training_history.history['val_accuracy']
recall = training_history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Training accuracy')
plt.plot(epochs_range, train_loss, label='Training Loss')

plt.title('Training')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(epochs_range, precision, label='Validation accuracy')
plt.plot(epochs_range, recall, label='Validation Loss')

plt.title('Validation')
plt.legend()

# In[ ]:

score = model.evaluate(x_test, y_test)
score_df = pd.Series(score, index=model.metrics_names)
score_df

# In[ ]:

y_pred = model.predict(x_test)
y_pred[:10]

# In[ ]:

y_pred = np.where(y_pred >= 0.5, 1, y_pred)
y_pred = np.where(y_pred < 0.5, 0, y_pred)

# In[ ]:

y_pred[:10]

# In[ ]:

y_test[:10]

# In[ ]:

accuracy_score(y_test, y_pred)
