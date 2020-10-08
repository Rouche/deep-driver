# coding: utf-8


import tensorflow as tf

from IPython import get_ipython

print(tf.__version__)

tf.debugging.set_log_device_placement(True)

x0 = tf.constant(3)

x0
print(x0)

x0.dtype

v1 = tf.Variable([[1.5, 2, 5], [2, 6, 8]])

v1

v2 = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

tf.add(v1, v2)

tf.convert_to_tensor(v1)

v1

v1[0, 0].assign(100)

v1.assign_add([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

print(v1.numpy())

v1.assign_sub([[4, 4, 4], [4, 4, 4]])

print(v1.numpy())

tf.compat.v1.executing_eagerly()

a = tf.constant(5, name="a")
b = tf.constant(8, name="b")

c = tf.add(a, b, name="sum")

sess = tf.compat.v1.Session()

# TensorFlow V1

# This will make an error because session does not have static model.
# sess.run(c)

# Turn off eager execution
tf.compat.v1.disable_eager_execution()
tf.compat.v1.executing_eagerly()

tf.compat.v1.reset_default_graph()

# now those computation wont be done yet.
a = tf.constant(5, name="a")
b = tf.constant(8, name="b")

c = tf.add(a, b, name="sum")

sess = tf.compat.v1.Session()
sess.run(c)

d = tf.multiply(a, b, name='product')
print(d)

sess.close()

m = tf.Variable([4.0, 5.0, 6.0], dtype=tf.float32, name='m')
c = tf.Variable([6.0, 7.0, 8.0], dtype=tf.float32, name='c')
m
c

x = tf.compat.v1.placeholder(tf.float32, shape=[3], name='x')
x

y = m * x + c
y

init = tf.compat.v1.global_variables_initializer()


get_ipython().system('rm -rf ./logs/')

with tf.compat.v1.Session() as sess:
    sess.run(init)

    y_output = sess.run(y, feed_dict={x: [100, 100, 100]})

    print("result = ", y_output)

    writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)
    writer.close()

get_ipython().run_line_magic('load_ext', 'tensorboard')

get_ipython().run_line_magic('tensorboard', '--logdir="./logs" --port 6060')
