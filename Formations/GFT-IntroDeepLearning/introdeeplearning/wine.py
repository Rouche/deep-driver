

import pandas as pd
import numpy as np
import tensorflow as tf
from enum import Enum

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers

#%%

def getData():
    data = pd.read_csv('./data/wine.data')
    x_set = data.iloc[:,1:14]
    y_set = data[data.columns[0]]
    print(x_set)
    print(y_set)

    x, x_test,y, y_test = train_test_split(x_set,y_set,test_size=0.33)

    return x.to_numpy() ,y.to_numpy(), x_test.to_numpy(),y_test.to_numpy()

#%% md

# Additional tools to utilize in solution:


#  https://keras.io/api/layers/activations/ - other activation functions (relu), softmax to nicely normalize output into probability wector
#  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout - dropout, clear information ocasionally to prevent overfitting
#  https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range - input normalisation so that every property has the same potential impact
#  https://keras.io/api/losses/ other loss functions (CategoricalCrossentropy)


#%% md

# If you have some spared time  there are some interesting links:
# - https://keras.io/layers/convolutional/  - way to go when working with images
# - https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5 - the way to prevent from overfitting
# - https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029 -normalization

num_classes = 4
epochs = 200
batch_size = 1

#%%

model = Sequential()

model.add(Dense(100,input_shape=(13,),activation='sigmoid', name='WineLayer'))
model.add(Dense(110,activation='sigmoid'))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

#%%

from tensorflow.keras.utils import plot_model
plot_model(model, to_file="./generated/model_wine.png", show_shapes=True, show_layer_names=True)

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

sample = np.array([[4.9,3.0,1.4,0.2,  5, 6, 7, 8, 9, 1, 23, 4.2, 1.9]])
result = model.predict(sample)

print(result)
