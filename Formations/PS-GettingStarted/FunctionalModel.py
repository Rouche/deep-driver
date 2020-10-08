#!/usr/bin/env python
# coding: utf-8

# In[3]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

# In[32]:

# dataset: https://www.kaggle.com/ronitf/heart-disease-uci
df = pd.read_csv('./datasets/heart.csv')
df.sample(5)

# In[33]:

df.shape

# In[34]:

df.isna().sum()

# In[35]:

df.describe().T

# In[36]:

df['sex'].value_counts()

# In[37]:

df['cp'].value_counts()

# In[40]:

sns.countplot(x=df['sex'], hue='target', data=df)

plt.title('Heart Disease Frequency for Gender')
plt.legend(["No Disease", "Yes Disease"])

plt.xlabel('Gender (0 = Female, 1 = Male')
plt.ylabel('Frequency')

plt.show()

# In[42]:

plt.figure(figsize=(20, 8))
sns.countplot(x=df['age'], hue='target', data=df)

plt.title('Heart Disease Frequency for Age')
plt.legend(["No Disease", "Yes Disease"])

plt.xlabel('Age')
plt.ylabel('Frequency')

plt.show()

# In[43]:

plt.figure(figsize=(10, 8))

plt.scatter(df['age'], df['chol'], s=200)

plt.xlabel('Age')
plt.ylabel('Cholesterol')

plt.show()

# In[64]:

features = df.drop('target', axis=1)
target = df[['target']]

# In[65]:

features.sample(5)

# In[66]:

target.sample(10)

# In[67]:

categorical_features = features[['sex', 'fbs', 'exang', 'cp', 'ca', 'slope', 'thal', 'restecg']].copy()
categorical_features.head()

# In[68]:

numeric_features = features[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].copy()
numeric_features.head()

# In[69]:

standardScaler = StandardScaler()

numeric_features = pd.DataFrame(standardScaler.fit_transform(numeric_features),
                                columns=numeric_features.columns,
                                index=numeric_features.index)
numeric_features.describe()

# In[70]:

processed_features = pd.concat([numeric_features, categorical_features], axis=1, sort=False)
processed_features.head()

# In[73]:

x_train, x_test, y_train, y_test = train_test_split(processed_features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=1)

x_train, x_val, y_train, y_val = train_test_split(processed_features,
                                                  target,
                                                  test_size=0.2,
                                                  random_state=1)
# In[75]:

print((x_train.shape, x_val.shape, x_test.shape))

# In[76]:

print((y_train.shape, y_val.shape, y_test.shape))


# In[77]:

def build_model():
    inputs = tf.keras.Input(shape=(x_train.shape[1],))

    dense_layer1 = layers.Dense(12, activation='relu')
    x = dense_layer1(inputs)

    dropout_layer = layers.Dropout(0.3)
    x = dropout_layer(x)

    dense_layer2 = layers.Dense(8, activation='relu')
    x = dense_layer2(x)

    predictions_layer = layers.Dense(1, activation='sigmoid')
    predictions = predictions_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # BinaryCrossentropy because we classify 1 disease 0 no disease. So its a binary loss.
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(0.5),
                           tf.keras.metrics.Recall(0.5)])
    return model


# In[78]:

model = build_model()

# In[79]:

dataset_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
dataset_train = dataset_train.batch(16)

dataset_train.shuffle(128)

# In[80]:

num_epochs = 100

# In[81]:

dataset_val = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))
dataset_val = dataset_val.batch(16)

# In[83]:

training_history = model.fit(dataset_train, epochs=num_epochs, validation_data=dataset_val)

# In[85]:

training_history.history.keys()

# In[86]:

train_acc = training_history.history['accuracy']
train_loss = training_history.history['loss']

precision = training_history.history['precision']
recall = training_history.history['recall']

epochs_range = range(num_epochs)

plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Training accuracy')
plt.plot(epochs_range, train_loss, label='Training Loss')

plt.title('Accuracy and Loss')

plt.subplot(1, 2, 2)

plt.plot(epochs_range, precision, label='Precision')
plt.plot(epochs_range, recall, label='Recall')

plt.title('Precision and Recall')
plt.legend()

# In[87]:

score = model.evaluate(x_test, y_test)
score_df = pd.Series(score, index=model.metrics_names)
score_df

# In[88]:

y_pred = model.predict(x_test)
y_pred[:10]

# In[89]:

y_pred = np.where(y_pred >= 0.5, 1, y_pred)
y_pred = np.where(y_pred < 0.5, 0, y_pred)

# In[90]:

y_pred[:10]

# In[91]:

pred_results = pd.DataFrame({'y_test': y_test.values.flatten(),
                             'y_pred': y_pred.flatten().astype('int32')}, index=range(len(y_pred)))

# In[97]:

pred_results.sample(10)

# In[98]:

pd.crosstab(pred_results.y_pred, pred_results.y_test)
