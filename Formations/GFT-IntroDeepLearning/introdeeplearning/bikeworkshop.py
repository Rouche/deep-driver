# Import the dependencies you will need in this exercise
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Import TensorFlow
import tensorflow as tf

# Avoid TF2 warning about float64->float32
tf.keras.backend.set_floatx('float64')
# Increase default plot size
matplotlib.rcParams['figure.figsize'] = (15.0, 8.0)


# Load and prepare data
# The first step when working with neural networks is preparing the data. This step can be more important than the neural network architecture or how the training is done.
#
# For instance, variables on different scales make it difficult for the network to learn efficiently.
#
# Let's start loading the data and inspecting it.

# Complete dataset
data_path = "./data/hour.csv"

rides = pd.read_csv(data_path)

# Reviewing the data
# This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split
# between casual and registered, summed up in the cnt column. You can see the first few rows of the data above.
#
# Below is a plot showing the number of bike riders over the first 10 days or so in the data set.
# (Some days don't have exactly 24 entries in the data set, so it's not exactly 10 days.)
# You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over
# all ridership and there are spikes when people are biking to and from work during the week. Looking
# at the data above, we also have information about temperature, humidity, and windspeed, all of these
# likely affecting the number of riders. You'll be trying to capture all this with your model.


rides[:24 * 10].plot(x='dteday', y='cnt')



rides.describe()

# Dummy variables
# Here we have some categorical variables like season, weather, month. To include these in our model,
# we'll need to make binary dummy variables. This is simple to do with Pandas thanks to get_dummies().

data = rides.copy()
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
    data = pd.concat([data, dummies], axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = data.drop(fields_to_drop, axis=1)
data.head()


# Pandas provide very interesting tools to better understand the data. Some of the are really easy to use!

# Scaling target variables
# To make training the network easier, we'll standardize each of the continuous variables.
# That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.
#
# The scaling factors are saved so we can go backwards when we use the network for predictions.


quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

data.head()

# Splitting the data into training, testing, and validation sets
# We'll save the data for the last approximately 21 days to use as a test set after we've trained the network.
# We'll use this set to make predictions and compare them with the actual number of riders.

# Save data for approximately the last 21 days
test_data = data[-21*24:]

# Now remove the test data from the data set
data = data[:-21*24]

# Separate the data into features and targets
# We create split each dataset in two, one with the input features and another with the target output

target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Training and validation data
# We'll split the data into two sets, one for training and one for validating as the network
# is being trained. Since this is time series data, we'll train on historical data,
# then try to predict on future data (the validation set). However, we won't do it this explicitly as we can let tensorflow do the split for us

# Hold out the last 60 days or so of the remaining data as a validation set
# train_features, train_targets = features[:-60*24], targets[:-60*24]
# val_features, val_targets = features[-60*24:], targets[-60*24:]

# Building the network
# Not it is time to build your network. In order to avoid tons of technical details,
# we will use the Keras API that comes with TensorFlow. Keras provides a high level API for building,
# training, exporting and importing deep neural networks.
#
# We will try to predict to total number of ride shares cnt.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(56, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1),
]
)

model.compile(
    # Optimizer
    optimizer = tf.keras.optimizers.SGD(),
    # Loss function to minimize
    loss= 'mse',
)


history = model.fit(features, targets['cnt'], validation_split=0.1, epochs=20)

model.summary()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.evaluate(test_features,  test_targets['cnt'], verbose=2)

# First we make the predictions and scale them
scaled_test_predictions = model.predict(test_features) * scaled_features['cnt'][1] + scaled_features['cnt'][0]

# And we plot the predictions against the real values
test_rides = rides[-21*24:]
plot_data = test_rides[['dteday', 'cnt']].copy()
plot_data['predicted'] = scaled_test_predictions.flatten()
plot_data.plot(x='dteday',y=['cnt','predicted'])
