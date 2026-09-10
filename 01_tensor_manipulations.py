"""
CS5720 Neural Network and Deep Learning
Home Assignment 1 - Part II, Task 1
Tensor Manipulations & Reshaping

This program demonstrates TensorFlow tensor creation, rank/shape,
reshaping, transposing, broadcasting, and addition.
"""

import tensorflow as tf

# 1. Create a random tensor with shape (4, 6).
# A fixed seed makes the output reproducible when we run the program again.
tf.random.set_seed(42)
tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.float32)

print("Original tensor:")
print(tensor.numpy())

# 2. Find the rank and shape using TensorFlow functions.
print("\nBefore reshaping/transposing:")
print("Rank:", tf.rank(tensor).numpy())
print("Shape:", tensor.shape)

# 3. Reshape (4, 6) -> (2, 3, 4), then transpose -> (3, 2, 4).
reshaped = tf.reshape(tensor, (2, 3, 4))
transposed = tf.transpose(reshaped, perm=[1, 0, 2])

print("\nAfter reshaping:")
print("Rank:", tf.rank(reshaped).numpy())
print("Shape:", reshaped.shape)

print("\nAfter transposing:")
print("Rank:", tf.rank(transposed).numpy())
print("Shape:", transposed.shape)

# 4. Broadcast a smaller tensor (1, 4) and add it to the (3, 2, 4) tensor.
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])

result = transposed + small_tensor

print("\nSmall tensor shape:", small_tensor.shape)
print("Transposed tensor shape:", transposed.shape)
print("Result shape after broadcasting and addition:", result.shape)

print("\nResult:")
print(result.numpy())

# 5. Explanation:
# TensorFlow broadcasts the (1, 4) tensor across the first two dimensions
# of the (3, 2, 4) tensor. The final dimension is already 4, so the
# addition can be performed element by element without manually copying
# the smaller tensor.
print("\nBroadcasting explanation:")
print(
    "The (1, 4) tensor is automatically expanded across the first two "
    "dimensions to behave like a (3, 2, 4) tensor during addition."
)
