"""
CS5720 Neural Network and Deep Learning
Home Assignment 1 - Part II, Task 4
Train a Neural Network and Log to TensorBoard

The model trains for 5 epochs and writes TensorBoard logs to:
logs/fit/
"""

from pathlib import Path
from datetime import datetime

import tensorflow as tf

# Load and preprocess MNIST.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create a simple neural network.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Create a unique run folder inside logs/fit/.
log_dir = Path("logs/fit") / datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=str(log_dir),
    histogram_freq=1
)

print("TensorBoard log directory:", log_dir)

# Train for exactly 5 epochs as required by the assignment.
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
    verbose=1
)

# Evaluate the model.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print("\nTensorBoard logs were saved inside logs/fit/")
print("Run this command in the project folder to open TensorBoard:")
print("tensorboard --logdir logs/fit")
