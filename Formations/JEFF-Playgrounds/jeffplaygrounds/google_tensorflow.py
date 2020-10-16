import tensorflow as tf

print('##################################')
print('#####        Starting        #####')
print('##################################')

x = tf.constant(3)
print(x.shape)

x = tf.constant([1, 2, 3])
print(x.shape)

x = tf.constant([[1, 2, 3], [1, 2, 3]])
print(x.shape)

x = tf.constant([[[1, 2, 3], [1, 2, 3]],
                 [[1, 2, 3], [1, 2, 3]]])
print(x.shape)

x1 = tf.constant([1, 2, 3])
x2 = tf.stack([x1, x1])
x3 = tf.stack([x2, x2, x2])
x4 = tf.stack([x3, x3])
print(x4.shape)

y = tf.constant([[1, 12, 3], [1, 32, 3]])
y1 = y[:, 1]
print(y1)

y2 = tf.reshape(y, (3, 2))
print(y2)

y.value_index(10.0)

def compute_gradient(u, v, w0, w1):
    with tf.GradientTape() as tape:
        loss = tf.losses.loss_mse(u, v, w0, w1)
    return tape.gradient(loss, [w0, w1])


w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dw0, dw1 = compute_gradient(u, v, w0, w1)
