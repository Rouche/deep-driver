import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Constants
WIDTH = 200
HEIGHT = 88

# For distributed environment with multiple dev sharing same resources.
# To not overload the server
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
load_previous = None
# Uncomment to train with another dataset
# load_previous = "models/workshop_model.20201007_1150_npy.0.90-0.13.h5"

dataset = "20201007_2322_npy"
directory = f"./data/{dataset}"

inputs_file = open(directory + "/inputs.npy", "br")
outputs_file = open(directory + "/outputs.npy", "br")

inputs = []
outputs = []
while True:
    try:
        input = np.load(inputs_file)
        inputs.append(input)
    except:
        break
while True:
    try:
        output = np.load(outputs_file)
        outputs.append(output)
    except:
        break

# with strategy.scope():
input_np = np.array(inputs)
output_np = np.array(outputs)

inputs = None
outputs = None
inputs_file.close()
outputs_file.close()
input_np = input_np[100:, :, :]
output_np = output_np[100:, :2]

print(input_np.shape)
print(output_np.shape)

# with strategy.scope():
x_train, x_test, y_train, y_test = train_test_split(input_np, output_np)

# with strategy.scope():
if load_previous is not None:
    model = models.load_model(load_previous)
else:
    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2))

model.summary()

# with strategy.scope():
if load_previous is None:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

my_callback = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(filepath="models/model." + dataset + ".{epoch:02d}.{val_accuracy:.2f}-{val_loss:.2f}.cp"),
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
]

# with strategy.scope():
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=my_callback)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_loss, test_acc)
model_file = f"models/workshop_model.{dataset}.{test_acc:.2f}-{test_loss:.2f}.h5"
model.save(model_file)
print(model_file)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
