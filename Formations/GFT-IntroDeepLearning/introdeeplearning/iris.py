
#%%

import pandas as pd
import numpy as np
from enum import Enum

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers


#%%

def IrisNameToInt(name):
    if name == 'Iris-setosa':
        return 1
    if name == 'Iris-versicolor':
        return 2
    if name == 'Iris-virginica':
        return 3

#%% md

# You can also download data from https://archive.ics.uci.edu/ml/datasets/iris

#%%

iris = pd.read_csv('./data/iris.data',header=None,prefix='COL')
iris

#%%

def getData():
    iris = pd.read_csv('./data/iris.data',header=None,prefix='COL')
    iris

    for i, val in enumerate(iris['COL4']):
        iris.at[i,'COL4'] = IrisNameToInt(iris.at[i,'COL4'])

    x_set = iris.iloc[:,0:4]
    y_set = iris.iloc[:,-1]

    # lets split data into train and test data
    x, x_test,y, y_test = train_test_split(x_set,y_set,test_size=0.33)

    return x.to_numpy() ,y.to_numpy(), x_test.to_numpy(),y_test.to_numpy()

#%%

num_classes = 4
epochs = 200
batch_size = 1

#%%

model = Sequential()

model.add(Dense(100,input_shape=(4,),activation='sigmoid', name='MyTestLayer'))
model.add(Dense(110,activation='sigmoid'))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

#%%

from tensorflow.keras.utils import plot_model
plot_model(model, to_file="./generated/model.png", show_shapes=True, show_layer_names=True)

#%%

model.compile(loss=losses.mean_squared_error, optimizer=optimizers.SGD(learning_rate=0.01),metrics=['accuracy'])

#%%

x_train,y_train,x_test,y_test = getData()
y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_test = tf.keras.utils.to_categorical(y_test,num_classes)


#%%

print(y_train)


#%%

history = model.fit(x_train,y_train,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)

#%%

import matplotlib.pyplot as plt


# Plot training accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

#%%

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

#%%

validate = model.evaluate(x_test,y_test)
print("\n\nTest loss: ",validate[0], "Test accuracy: ", validate[1])

#%%

sample = np.array([[4.9,3.0,1.4,0.2]])
result = model.predict(sample)

print(result)
