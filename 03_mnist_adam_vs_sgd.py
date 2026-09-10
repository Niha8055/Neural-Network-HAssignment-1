"""
CS5720 Neural Network and Deep Learning
Home Assignment 1 - Part II, Task 3
Train MNIST Model with Adam & SGD

This program trains two similar neural networks:
1. One using Adam
2. One using SGD

It compares their training and validation accuracy.
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# Make the experiment repeatable.
tf.random.set_seed(42)

# 1. Load the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

def build_model():
    """Create the same neural network architecture for a fair comparison."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

# 2. Train Model A with Adam.
adam_model = build_model()
adam_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Adam model...")
adam_history = adam_model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 3. Train Model B with SGD.
sgd_model = build_model()
sgd_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining SGD model...")
sgd_history = sgd_model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate both models on the test set.
adam_test_loss, adam_test_accuracy = adam_model.evaluate(x_test, y_test, verbose=0)
sgd_test_loss, sgd_test_accuracy = sgd_model.evaluate(x_test, y_test, verbose=0)

print("\nFinal test results:")
print(f"Adam test accuracy: {adam_test_accuracy:.4f}")
print(f"SGD test accuracy:  {sgd_test_accuracy:.4f}")

# 4. Compare training and validation accuracy trends.
epochs = range(1, 6)

plt.figure(figsize=(9, 5))
plt.plot(epochs, adam_history.history["accuracy"], marker="o", label="Adam - Training")
plt.plot(epochs, adam_history.history["val_accuracy"], marker="o", label="Adam - Validation")
plt.plot(epochs, sgd_history.history["accuracy"], marker="s", label="SGD - Training")
plt.plot(epochs, sgd_history.history["val_accuracy"], marker="s", label="SGD - Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy: Adam vs. SGD")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("adam_vs_sgd_accuracy.png", dpi=150)
plt.show()

print("\nPlot saved as: adam_vs_sgd_accuracy.png")
