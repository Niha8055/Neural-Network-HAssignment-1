# CS5720 Neural Network and Deep Learning - Home Assignment 1

**University:** University of Central Missouri  
**Department:** Computer Science & Cybersecurity  
**Course:** CS5720 Neural Network and Deep Learning  
**Semester:** Fall 2026  
**Student Name:** NIHARIKA MANNEPALLI  
**Student ID:** 700#775394 

## Assignment Overview

This repository contains my source code for Home Assignment 1. The programming part covers:

1. Tensor Manipulations & Reshaping
2. Loss Functions & Hyperparameter Tuning
3. Training an MNIST Model with Adam and SGD
4. Training an MNIST Model and logging the results to TensorBoard

---

# Part I - Short Answers

## Question 1

### a. Traditional programming vs. machine learning

In traditional programming, the programmer writes explicit rules that tell the computer how to solve a problem. In machine learning, we give the computer data and the expected results, and the algorithm learns patterns from the data to make predictions on new data.

### b. Relationship among AI, ML, and DL

Artificial Intelligence is the broad field of making computers perform tasks that normally require human intelligence. Machine Learning is a part of AI where systems learn patterns from data instead of relying only on explicitly written rules. Deep Learning is a part of machine learning that uses neural networks with multiple layers to learn complex patterns.

### c. Two reasons deep learning became more successful

1. Large datasets became available for training models.
2. More powerful hardware, especially GPUs, made it practical to train large neural networks.

---

## Question 2

### a. Input, hidden, and output layers

The input layer receives the features or data given to the neural network. Hidden layers transform the input by applying weights, biases, and activation functions to learn useful patterns. The output layer produces the final prediction or result.

### b. Weights and biases

Weights determine how strongly each input affects a neuron. A bias shifts the neuron's weighted sum and helps the neuron fit the data better. A neuron calculates a weighted sum such as:

`z = w1*x1 + w2*x2 + ... + b`

The result is then passed through an activation function.

### c. Why an activation function is needed

An activation function introduces non-linearity into a neural network. Without activation functions, multiple neural-network layers would behave like one linear operation, making it difficult for the network to learn complex relationships.

---

## Question 3

### a. What is a perceptron?

A perceptron is a simple artificial neuron used for binary classification. It calculates a weighted sum of its inputs plus a bias and applies a threshold. If the result reaches the threshold, it produces 1; otherwise it produces 0.

### b. Why can one perceptron solve AND and OR?

AND and OR are linearly separable problems. A single perceptron can find a decision boundary that separates the positive and negative examples for these two logical operations.

### c. Why cannot one perceptron solve XOR?

XOR is not linearly separable, so one straight decision boundary cannot separate its classes correctly. A multilayer neural network solves this by combining multiple neurons and nonlinear activation functions to create more complex decision boundaries.

---

## Question 4

### a. Sigmoid, Tanh, and ReLU

- **Sigmoid:** output is between 0 and 1. It is useful when an output can be interpreted as a probability.
- **Tanh:** output is between -1 and 1 and is centered around zero.
- **ReLU:** outputs 0 for negative inputs and approximately the input value for positive inputs. It is widely used in hidden layers.

### b. Vanishing-gradient problem and ReLU

The vanishing-gradient problem happens when gradients become extremely small during backpropagation, making earlier layers learn very slowly. Sigmoid and Tanh can produce very small gradients when they are saturated. ReLU helps reduce this problem for positive inputs because its gradient is 1.

### c. Neural-network training cycle

1. **Forward propagation:** input data passes through the network to produce predictions.
2. **Loss calculation:** the prediction is compared with the true answer to calculate the error/loss.
3. **Backpropagation:** gradients are calculated to determine how the weights contributed to the error.
4. **Weight update:** the optimizer uses the gradients to update the weights and biases so the model can improve.

---

# Part II - Programming

## 1. Tensor Manipulations & Reshaping

### File

`01_tensor_manipulations.py`

### What the program does

- Creates a random `(4, 6)` tensor.
- Prints its rank and shape.
- Reshapes it to `(2, 3, 4)`.
- Transposes it to `(3, 2, 4)`.
- Adds a `(1, 4)` tensor using broadcasting.
- Prints the resulting shape and values.

### Expected shapes

```text
Original:    (4, 6)
Reshaped:    (2, 3, 4)
Transposed:  (3, 2, 4)
Small tensor: (1, 4)
Result:      (3, 2, 4)
```

### Broadcasting explanation

TensorFlow can automatically expand dimensions when tensor shapes are compatible. Here, the `(1, 4)` tensor is reused across the first two dimensions of the `(3, 2, 4)` tensor, so we do not have to manually copy it.

---

## 2. Loss Functions & Hyperparameter Tuning

### File

`02_loss_functions.py`

The program:

- Defines one-hot `y_true` values.
- Defines original model predictions.
- Defines slightly modified predictions.
- Calculates Mean Squared Error (MSE).
- Calculates Categorical Cross-Entropy (CCE).
- Prints the losses and their changes.
- Creates `loss_comparison.png`.

The modified predictions are less confident on the correct classes, so the loss should increase.

---

## 3. Train MNIST with Adam and SGD

### File

`03_mnist_adam_vs_sgd.py`

The program:

- Loads MNIST.
- Normalizes pixel values from 0-255 to 0-1.
- Creates the same neural-network architecture for both experiments.
- Trains one model with Adam.
- Trains another model with SGD.
- Uses 5 epochs.
- Compares training and validation accuracy.
- Creates `adam_vs_sgd_accuracy.png`.

The architecture is intentionally kept the same so that the main difference in the experiment is the optimizer.

---

## 4. TensorBoard

### File

`04_tensorboard_mnist.py`

The program:

- Loads and preprocesses MNIST.
- Creates a simple neural network.
- Uses Adam as the optimizer.
- Trains for exactly 5 epochs.
- Saves TensorBoard information in `logs/fit/`.

To launch TensorBoard:

```bash
tensorboard --logdir logs/fit
```

Then open the local address displayed in the terminal, usually something similar to:

```text
http://localhost:6006/
```

### Question 1: What patterns do you observe?

Training accuracy generally increases as the model learns. Validation accuracy should also improve if the model is learning patterns that generalize to unseen data. Small differences between the two curves are normal.

### Question 2: How can TensorBoard detect overfitting?

If training accuracy continues increasing while validation accuracy stops improving or decreases, the model may be overfitting. Similarly, if training loss keeps decreasing while validation loss starts increasing, this is a strong sign of overfitting.

### Question 3: What happens when you increase the number of epochs?

The model gets more opportunities to learn from the training data. At first, this can improve accuracy, but too many epochs can cause overfitting, where the model becomes very good on training data but performs worse on validation data.

---

# How to Run the Project

## Step 1 - Install Python

Install Python 3.10 or another TensorFlow-compatible Python version if Python is not already installed.

Check your Python version:

```bash
python --version
```

## Step 2 - Create a virtual environment

Open Command Prompt or PowerShell inside this project folder:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

## Step 3 - Install the required packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4 - Run Task 1

```bash
python 01_tensor_manipulations.py
```

## Step 5 - Run Task 2

```bash
python 02_loss_functions.py
```

This creates:

```text
loss_comparison.png
```

## Step 6 - Run Task 3

```bash
python 03_mnist_adam_vs_sgd.py
```

This creates:

```text
adam_vs_sgd_accuracy.png
```

## Step 7 - Run Task 4

```bash
python 04_tensorboard_mnist.py
```

Then run:

```bash
tensorboard --logdir logs/fit
```

Open the local TensorBoard address shown in the terminal.

---

# Suggested Repository Structure

```text
CS5720_Home_Assignment_1/
│
├── 01_tensor_manipulations.py
├── 02_loss_functions.py
├── 03_mnist_adam_vs_sgd.py
├── 04_tensorboard_mnist.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── loss_comparison.png
└── adam_vs_sgd_accuracy.png
```

Do not upload the `logs/` directory because it is excluded by `.gitignore`. The assignment asks you to create the logs locally for TensorBoard analysis.

---

# GitHub Submission Steps

## Step 1 - Create a GitHub repository

Go to GitHub and sign in.

Create a new repository named:

```text
CS5720-Home-Assignment-1
```

Choose **Private** or **Public** according to your course requirement.

Do not add another README if you already have the README in this project folder.

## Step 2 - Open the project folder

Open the folder containing the Python files.

Right-click inside the folder and choose **Open in Terminal**.

## Step 3 - Configure Git

Run:

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR GITHUB EMAIL"
```

## Step 4 - Initialize the repository

```bash
git init
```

## Step 5 - Add all files

```bash
git add .
```

Check what will be committed:

```bash
git status
```

## Step 6 - Create your first commit

```bash
git commit -m "Complete CS5720 Home Assignment 1"
```

## Step 7 - Connect the local folder to GitHub

On your GitHub repository page, copy the repository HTTPS address.

It will look similar to:

```text
https://github.com/YOUR-USERNAME/CS5720-Home-Assignment-1.git
```

Then run:

```bash
git remote add origin https://github.com/YOUR-USERNAME/CS5720-Home-Assignment-1.git
```

## Step 8 - Rename the branch to main

```bash
git branch -M main
```

## Step 9 - Push the project

```bash
git push -u origin main
```

If GitHub asks you to authenticate, complete the GitHub sign-in/authentication process.

---

# Final GitHub Check

Before submitting Brightspace, open your GitHub repository in the browser and make sure you can see:

- `README.md`
- `01_tensor_manipulations.py`
- `02_loss_functions.py`
- `03_mnist_adam_vs_sgd.py`
- `04_tensorboard_mnist.py`
- `requirements.txt`
- `.gitignore`
- `loss_comparison.png`
- `adam_vs_sgd_accuracy.png`
