"""
CS5720 Neural Network and Deep Learning
Home Assignment 1 - Part II, Task 2
Loss Functions & Hyperparameter Tuning

This program calculates MSE and Categorical Cross-Entropy (CCE),
changes the predictions slightly, and plots the loss values.
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# 1. True values: three samples and three possible classes.
y_true = tf.constant([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# Original model predictions.
y_pred_original = tf.constant([
    [0.70, 0.20, 0.10],
    [0.10, 0.80, 0.10],
    [0.20, 0.20, 0.60]
])

# Slightly modified predictions.
# The correct class probabilities are lower here, so the model is
# making less confident/correct predictions.
y_pred_modified = tf.constant([
    [0.60, 0.25, 0.15],
    [0.15, 0.70, 0.15],
    [0.25, 0.25, 0.50]
])

mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

# 2. Compute MSE and CCE for the original predictions.
mse_original = mse(y_true, y_pred_original).numpy()
cce_original = cce(y_true, y_pred_original).numpy()

# 3. Compute the losses after modifying the predictions.
mse_modified = mse(y_true, y_pred_modified).numpy()
cce_modified = cce(y_true, y_pred_modified).numpy()

print("Original predictions:")
print("MSE:", round(mse_original, 4))
print("Categorical Cross-Entropy:", round(cce_original, 4))

print("\nModified predictions:")
print("MSE:", round(mse_modified, 4))
print("Categorical Cross-Entropy:", round(cce_modified, 4))

print("\nChange in loss:")
print("MSE change:", round(mse_modified - mse_original, 4))
print("CCE change:", round(cce_modified - cce_original, 4))

# 4. Plot a bar chart comparing the losses.
loss_names = ["MSE", "Cross-Entropy"]
original_values = [mse_original, cce_original]
modified_values = [mse_modified, cce_modified]

x = range(len(loss_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width / 2 for i in x], original_values, width, label="Original")
plt.bar([i + width / 2 for i in x], modified_values, width, label="Modified")

plt.xticks(list(x), loss_names)
plt.ylabel("Loss value")
plt.title("MSE vs. Categorical Cross-Entropy")
plt.legend()
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=150)
plt.show()

print("\nPlot saved as: loss_comparison.png")
