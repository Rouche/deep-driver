#!/usr/bin/env python
# coding: utf-8

# # Introduction to Tensors and Variables
# 
# **Learning Objectives**
# 
# 1. Understand Basic and Advanced Tensor Concepts
# 2. Understand Single-Axis and Multi-Axis Indexing
# 3. Create Tensors and Variables
# 
# 
# 
# 
# ## Introduction 
# 
# In this notebook, we look at tensors, which are multi-dimensional arrays with a uniform type (called a dtype). You can see all supported dtypes at [tf.dtypes.DType](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType).  If you're familiar with [ NumPy](https://numpy.org/devdocs/user/quickstart.html), tensors are (kind of) like np.arrays.  All tensors are immutable like python numbers and strings: you can never update the contents of a tensor, only create a new one.
# 
# We also look at variables, a `tf.Variable` represents a tensor whose value can be changed by running operations (ops) on it.  Specific ops allow you to read and modify the values of this tensor. Higher level libraries like `tf.keras` use `tf.Variable` to store model parameters.

# ## Load necessary libraries 
# We will start by importing the necessary libraries for this lab.

import numpy as np
import tensorflow as tf

print("TensorFlow version: ",tf.version.VERSION)


# # Lab Task 1: Understand Basic and Advanced Tensor Concepts

# ## Basics
# 
# Let's create some basic tensors.

# Here is a "scalar" or "rank-0" tensor . A scalar contains a single value, and no "axes".

# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)


# A "vector" or "rank-1" tensor is like a list of values. A vector has 1-axis:

# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)


# A "matrix" or "rank-2" tensor has 2-axes:

# In[6]:


# If we want to be specific, we can set the dtype (see below) at creation time
# TODO 1a
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)


# <table>
# <tr>
#   <th>A scalar, shape: <code>[]</code></th>
#   <th>A vector, shape: <code>[3]</code></th>
#   <th>A matrix, shape: <code>[3, 2]</code></th>
# </tr>
# <tr>
#   <td>
#    <img src="../images/tensor/scalar.png" alt="A scalar, the number 4" />
#   </td>
# 
#   <td>
#    <img src="../images/tensor/vector.png" alt="The line with 3 sections, each one containing a number."/>
#   </td>
#   <td>
#    <img src="../images/tensor/matrix.png" alt="A 3x2 grid, with each cell containing a number.">
#   </td>
# </tr>
# </table>
# 

# Tensors may have more axes, here is a tensor with 3-axes:

# In[7]:


# There can be an arbitrary number of
# axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)


# There are many ways you might visualize a tensor with more than 2-axes.

# <table>
# <tr>
#   <th colspan=3>A 3-axis tensor, shape: <code>[3, 2, 5]</code></th>
# <tr>
# <tr>
#   <td>
#    <img src="../images/tensor/3-axis_numpy.png"/>
#   </td>
#   <td>
#    <img src="../images/tensor/3-axis_front.png"/>
#   </td>
# 
#   <td>
#    <img src="../images/tensor/3-axis_block.png"/>
#   </td>
# </tr>
# 
# </table>

# You can convert a tensor to a NumPy array either using `np.array` or the `tensor.numpy` method:

# In[8]
# TODO 1b
np.array(rank_2_tensor)

# TODO 1c
rank_2_tensor.numpy()


# Tensors often contain floats and ints, but have many other types, including:
# 
# * complex numbers
# * strings
# 
# The base `tf.Tensor` class requires tensors to be "rectangular"---that is, along each axis, every element is the same size.  However, there are specialized types of Tensors that can handle different shapes: 
# 
# * ragged (see [RaggedTensor](#ragged_tensors) below)
# * sparse (see [SparseTensor](#sparse_tensors) below)

# We can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication.

# In[9]:


a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")


# In[10]:


print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication


# Tensors are used in all kinds of operations (ops).

# In[11]:


c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# TODO 1d
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))


# ## About shapes

# Tensors have shapes.  Some vocabulary:
# 
# * **Shape**: The length (number of elements) of each of the dimensions of a tensor.
# * **Rank**: Number of tensor dimensions.  A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
# * **Axis** or **Dimension**: A particular dimension of a tensor.
# * **Size**: The total number of items in the tensor, the product shape vector
# 

# Note: Although you may see reference to a "tensor of two dimensions", a rank-2 tensor does not usually describe a 2D space.

# Tensors and `tf.TensorShape` objects have convenient properties for accessing these:

# In[12]:


rank_4_tensor = tf.zeros([3, 2, 4, 5])


# <table>
# <tr>
#   <th colspan=2>A rank-4 tensor, shape: <code>[3, 2, 4, 5]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="../images/tensor/shape.png" alt="A tensor shape is like a vector.">
#     <td>
# <img src="../images/tensor/4-axis_block.png" alt="A 4-axis tensor">
#   </td>
#   </tr>
# </table>
# 

# In[13]:


print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())


# While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.
# 
# <table>
# <tr>
# <th>Typical axis order</th>
# </tr>
# <tr>
#     <td>
# <img src="../images/tensor/shape2.png" alt="Keep track of what each axis is. A 4-axis tensor might be: Batch, Width, Height, Freatures">
#   </td>
# </tr>
# </table>

# ## Lab Task 2: Understand Single-Axis and Multi-Axis Indexing

# ### Single-axis indexing
# 
# TensorFlow follow standard python indexing rules, similar to [indexing a list or a string in python](https://docs.python.org/3/tutorial/introduction.html#strings), and the bacic rules for numpy indexing.
# 
# * indexes start at `0`
# * negative indices count backwards from the end
# * colons, `:`, are used for slices `start:stop:step`
# 

# In[14]:


rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())


# Indexing with a scalar removes the dimension:

# In[15]:


print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())


# Indexing with a `:` slice keeps the dimension:

# In[16]:


print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())


# ### Multi-axis indexing

# Higher rank tensors are indexed by passing multiple indices. 
# 
# The single-axis exact same rules as in  the single-axis case apply to each axis independently.

# In[17]:


print(rank_2_tensor.numpy())


# Passing an integer for each index the result is a scalar.

# In[18]:


# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())


# You can index using any combination integers and slices:

# In[19]:


# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")


# Here is an example with a 3-axis tensor:

# In[20]:


print(rank_3_tensor[:, :, 4])


# <table>
# <tr>
# <th colspan=2>Selecting the last feature across all locations in each example in the batch </th>
# </tr>
# <tr>
#     <td>
# <img src="../images/tensor/index1.png" alt="A 3x2x5 tensor with all the values at the index-4 of the last axis selected.">
#   </td>
#       <td>
# <img src="../images/tensor/index2.png" alt="The selected values packed into a 2-axis tensor.">
#   </td>
# </tr>
# </table>

# ## Manipulating Shapes
# 
# Reshaping a tensor is of great utility. 
# 
# The `tf.reshape` operation is fast and cheap as the underlying data does not need to be duplicated.
# 

# In[21]:


# Shape returns a `TensorShape` object that shows the size on each dimension
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)


# In[22]:


# You can convert this object into a Python list, too
print(var_x.shape.as_list())


# You can reshape a tensor into a new shape.  Reshaping is fast and cheap as the underlying data does not need to be duplicated.

# In[23]:


# TODO 2a
# We can reshape a tensor to a new shape.
# Note that we're passing in a list
reshaped = tf.reshape(var_x, [1, 3])


# In[24]:


print(var_x.shape)
print(reshaped.shape)


# The data maintains it's layout in memory and a new tensor is created, with the requested shape, pointing to the same data. TensorFlow uses C-style "row-major" memory ordering, where incrementing the right-most index corresponds to a single step in memory.

# In[25]:


print(rank_3_tensor)


# If you flatten a tensor you can see what order it is laid out in memory.

# In[26]:


# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))


# Typically the only reasonable uses of `tf.reshape` are to combine or split adjacent axes (or add/remove `1`s).
# 
# For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do, as the slices do not mix:

# In[27]:


print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))


# <table>
# <th colspan=3>
# Some good reshapes.
# </th>
# <tr>
#   <td>
# <img src="../images/tensor/reshape-before.png" alt="A 3x2x5 tensor">
#   </td>
#   <td>
#   <img src="../images/tensor/reshape-good1.png" alt="The same data reshaped to (3x2)x5">
#   </td>
#   <td>
# <img src="../images/tensor/reshape-good2.png" alt="The same data reshaped to 3x(2x5)">
#   </td>
# </tr>
# </table>
# 

# Reshaping will "work" for any new shape with the same total number of elements, but it will not do anything useful if you do not respect the order of the axes.
# 
# Swapping axes in `tf.reshape` does not work, you need `tf.transpose` for that. 
# 

# In[28]:

print(rank_3_tensor)
# Bad examples: don't do this

# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e: print(e)


# <table>
# <th colspan=3>
# Some bad reshapes.
# </th>
# <tr>
#   <td>
# <img src="../images/tensor/reshape-bad.png" alt="You can't reorder axes, use tf.transpose for that">
#   </td>
#   <td>
# <img src="../images/tensor/reshape-bad4.png" alt="Anything that mixes the slices of data together is probably wrong.">
#   </td>
#   <td>
# <img src="../images/tensor/reshape-bad2.png" alt="The new shape must fit exactly.">
#   </td>
# </tr>
# </table>

# You may run across not-fully-specified shapes. Either the shape contains a `None` (a dimension's length is unknown) or the shape is `None` (the rank of the tensor is unknown).
# 
# Except for [tf.RaggedTensor](#ragged_tensors), this will only occur in the context of TensorFlow's, symbolic, graph-building  APIs: 
# 
# * [tf.function](function.ipynb) 
# * The [keras functional API](keras/functional.ipynb).
# 

# ## More on `DTypes`
# 
# To inspect a `tf.Tensor`'s data type use the `Tensor.dtype` property.
# 
# When creating a `tf.Tensor` from a Python object you may optionally specify the datatype.
# 
# If you don't, TensorFlow chooses a datatype that can represent your data. TensorFlow converts Python integers to `tf.int32` and python floating point numbers to `tf.float32`. Otherwise TensorFlow uses the same rules NumPy uses when converting to arrays.
# 
# You can cast from type to type.

# In[29]:


# TODO 2b
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, let's cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)


# ## Broadcasting
# 
# Broadcasting is a concept borrowed from the [equivalent feature in NumPy](https://numpy.org/doc/stable/user/basics.html).  In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.
# 
# The simplest and most common case is when you attempt to multiply or add a tensor to a scalar.  In that case, the scalar is broadcast to be the same shape as the other argument. 

# In[30]:


x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)


# Likewise, 1-sized dimensions can be stretched out to match the other arguments.  Both arguments can be stretched in the same computation.
# 
# In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to produce a 3x4 matrix. Note how the leading 1 is optional: The shape of y is `[4]`.

# In[31]:


# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))


# <table>
# <tr>
#   <th>A broadcasted add: a <code>[3, 1]</code> times a <code>[1, 4]</code> gives a <code>[3,4]</code> </th>
# </tr>
# <tr>
#   <td>
# <img src="../images/tensor/broadcasting.png" alt="Adding a 3x1 matrix to a 4x1 matrix results in a 3x4 matrix">
#   </td>
# </tr>
# </table>
# 

# Here is the same operation without broadcasting:

# In[32]:


x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading


# Most of the time, broadcasting is both time and space efficient, as the broadcast operation never materializes the expanded tensors in memory.
# 
# You see what broadcasting looks like using `tf.broadcast_to`.

# In[33]:


print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))


# Unlike a mathematical op, for example, `broadcast_to` does nothing special to save memory.  Here, you are materializing the tensor.
# 
# It can get even more complicated.  [This section](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) of Jake VanderPlas's book _Python Data Science Handbook_ shows more broadcasting tricks (again in NumPy).

# ## tf.convert_to_tensor
# 
# Most ops, like `tf.matmul` and `tf.reshape` take arguments of class `tf.Tensor`.  However, you'll notice in the above case, we frequently pass Python objects shaped like tensors.
# 
# Most, but not all, ops call `convert_to_tensor` on non-tensor arguments.  There is a registry of conversions, and most object classes like NumPy's `ndarray`, `TensorShape`, Python lists, and `tf.Variable` will all convert automatically.
# 
# See `tf.register_tensor_conversion_function` for more details, and if you have your own type you'd like to automatically convert to a tensor.

# ## Ragged Tensors
# 
# A tensor with variable numbers of elements along some axis is called "ragged". Use `tf.ragged.RaggedTensor` for ragged data.
# 
# For example, This cannot be represented as a regular tensor:

# <table>
# <tr>
#   <th>A `tf.RaggedTensor`, shape: <code>[4, None]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="../images/tensor/ragged.png" alt="A 2-axis ragged tensor, each row can have a different length.">
#   </td>
# </tr>
# </table>

# In[34]:


ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]


# In[35]:


try:
  tensor = tf.constant(ragged_list)
except Exception as e: print(e)


# Instead create a `tf.RaggedTensor` using `tf.ragged.constant`:

# In[36]:


# TODO 2c
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)


# The shape of a `tf.RaggedTensor` contains unknown dimensions:

# In[37]:


print(ragged_tensor.shape)


# ## String tensors
# 
# `tf.string` is a `dtype`, which is to say we can represent data as strings (variable-length byte arrays) in tensors.  
# 
# The strings are atomic and cannot be indexed the way Python strings are. The length of the string is not one of the dimensions of the tensor. See `tf.strings` for functions to manipulate them.

# Here is a scalar string tensor:

# In[38]:


# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)


# And a vector of strings:

# <table>
# <tr>
#   <th>A vector of strings, shape: <code>[3,]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="../images/tensor/strings.png" alt="The string length is not one of the tensor's axes.">
#   </td>
# </tr>
# </table>

# In[39]:


# If we have two string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (2,), indicating that it is 2 x unknown.
print(tensor_of_strings)


# In the above printout the `b` prefix indicates that `tf.string` dtype is not a unicode string, but a byte-string. See the [Unicode Tutorial](https://www.tensorflow.org/tutorials/load_data/unicode) for more about working with unicode text in TensorFlow.

# If you pass unicode characters they are utf-8 encoded.

# In[40]:


tf.constant("🥳👍")


# Some basic functions with strings can be found in `tf.strings`, including `tf.strings.split`.

# In[41]:


# We can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))


# In[42]:


# ...but it turns into a `RaggedTensor` if we split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))


# <table>
# <tr>
#   <th>Three strings split, shape: <code>[3, None]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="../images/tensor/string-split.png" alt="Splitting multiple strings returns a tf.RaggedTensor">
#   </td>
# </tr>
# </table>

# And `tf.string.to_number`:

# In[43]:


text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))


# Although you can't use `tf.cast` to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.

# In[44]:


byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)


# In[45]:


# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)


# The `tf.string` dtype is used for all raw bytes data in TensorFlow. The `tf.io` module contains functions for converting data to and from bytes, including decoding images and parsing csv.

# ## Sparse tensors
# 
# Sometimes, your data is sparse, like a very wide embedding space.  TensorFlow supports `tf.sparse.SparseTensor` and related operations to store sparse data efficiently.

# <table>
# <tr>
#   <th>A `tf.SparseTensor`, shape: <code>[3, 4]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="../images/tensor/sparse.png" alt="An 3x4 grid, with values in only two of the cells.">
#   </td>
# </tr>
# </table>

# In[46]:


# Sparse tensors store values by index in a memory-efficient manner
# TODO 2d
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# We can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))


# # Lab Task 3: Introduction to Variables

# A TensorFlow **variable** is the recommended way to represent shared, persistent state your program manipulates. This guide covers how to create, update, and manage instances of `tf.Variable` in TensorFlow.
# 
# Variables are created and tracked via the `tf.Variable` class. A `tf.Variable` represents a tensor whose value can be changed by running ops on it.  Specific ops allow you to read and modify the values of this tensor. Higher level libraries like `tf.keras` use `tf.Variable` to store model parameters. 

# ## Setup
# 
# This notebook discusses variable placement.  If you want to see on what device your variables are placed, uncomment this line.

# In[ ]:


import tensorflow as tf

# Uncomment to see where your variables get placed (see below)
# tf.debugging.set_log_device_placement(True)


# ## Create a variable
# 
# To create a variable, provide an initial value.  The `tf.Variable` will have the same `dtype` as the initialization value.

# In[ ]:


# TODO 3a
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])


# A variable looks and acts like a tensor, and, in fact, is a data structure backed by a `tf.Tensor`.  Like tensors, they have a `dtype` and a shape, and can be exported to NumPy.

# In[ ]:


print("Shape: ",my_variable.shape)
print("DType: ",my_variable.dtype)
print("As NumPy: ", my_variable.numpy)


# Most tensor operations work on variables as expected, although variables cannot be reshaped.

# In[ ]:


print("A variable:",my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, ([1,4])))


# As noted above, variables are backed by tensors. You can reassign the tensor using `tf.Variable.assign`.  Calling `assign` does not (usually) allocate a new tensor; instead, the existing tensor's memory is reused.

# In[ ]:


a = tf.Variable([2.0, 3.0])
# This will keep the same dtype, float32
a.assign([1, 2])
# Not allowed as it resizes the variable: 
try:
  a.assign([1.0, 2.0, 3.0])
except Exception as e: print(e)


# If you use a variable like a tensor in operations, you will usually operate on the backing tensor.
# 
# Creating new variables from existing variables duplicates the backing tensors. Two variables will not share the same memory.

# In[ ]:


a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]


# ## Lifecycles, naming, and watching
# 
# In Python-based TensorFlow, `tf.Variable` instance have the same lifecycle as other Python objects. When there are no references to a variable it is automatically deallocated.
# 
# Variables can also be named which can help you track and debug them.  You can give two variables the same name.

# In[ ]:


# Create a and b; they have the same value but are backed by different tensors.
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")

# These are elementwise-unequal, despite having the same name
print(a == b)


# Variable names are preserved when saving and loading models. By default, variables in models will acquire unique variable names automatically, so you don't need to assign them yourself unless you want to.
# 
# Although variables are important for differentiation, some variables will not need to be differentiated.  You can turn off gradients for a variable by setting `trainable` to false at creation. An example of a variable that would not need gradients is a training step counter.

# In[ ]:


step_counter = tf.Variable(1, trainable=False)
print(step_counter)

# ## Placing variables and tensors
# 
# For better performance, TensorFlow will attempt to place tensors and variables on the fastest device compatible with its `dtype`.  This means most variables are placed on a GPU if one is available.
# 
# However, we can override this.  In this snippet, we can place a float tensor and a variable on the CPU, even if a GPU is available.  By turning on device placement logging (see [Setup](#scrollTo=xZoJJ4vdvTrD)), we can see where the variable is placed. 
# 
# Note: Although manual placement works, using [distribution strategies](distributed_training) can be a more convenient and scalable way to optimize your computation.
# 
# If you run this notebook on different backends with and without a GPU you will see different logging.  *Note that logging device placement must be turned on at the start of the session.*

# In[ ]:


with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)


# It's possible to set the location of a variable or tensor on one device and do the computation on another device.  This will introduce delay, as data needs to be copied between the devices.
# 
# You might do this, however, if you had multiple GPU workers but only want one copy of the variables.

# In[ ]:


with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)


# Note: Because `tf.config.set_soft_device_placement` is turned on by default, even if you run this code on a device without a GPU, it will still run and the multiplication step happen on the CPU.
# 
# For more on distributed training, see [our guide](distributed_training).

# ## Next steps
# 
# To understand how variables are typically used, see our guide on [automatic distribution](autodiff).

# Copyright 2020 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
