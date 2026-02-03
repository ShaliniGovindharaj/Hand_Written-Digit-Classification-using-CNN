# Hand_Written-Digit-Classification-using-CNN

Project Overview
This project implements a Neural Network (NN) to classify handwritten digits (0-9) using the MNIST dataset. It demonstrates the fundamental concepts of Deep Learning, including data preprocessing, model building, training, and evaluation using TensorFlow and Keras.

The model takes a 28x28 grayscale image of a handwritten digit as input and predicts the corresponding digit with high accuracy.
---
Dataset
The project uses the classic MNIST (Modified National Institute of Standards and Technology) dataset.

Training Set: 60,000 images of handwritten digits.
Test Set: 10,000 images.
Image Size: 28x28 pixels (grayscale).
Labels: Integers from 0 to 9.
---
Workflow
Data Loading: Importing the MNIST dataset directly from keras.datasets.
Preprocessing:
Normalization: Scaling pixel values from 0-255 to 0-1 to improve training convergence.
Flattening: Converting 2D images (28x28) into 1D arrays (784 features) for the input layer.
---
Model Architecture:
Input Layer: Flattens the 28x28 input.
Hidden Layers: Dense layers with ReLU activation.
Output Layer: Dense layer with 10 neurons (for digits 0-9) using Softmax or Sigmoid activation.
----
Training:
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
Evaluation: Testing the model on unseen data and plotting the Confusion Matrix.
Model Architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input Layer
    keras.layers.Dense(50, activation='relu'),   # Hidden Layer
    keras.layers.Dense(50, activation='relu'),   # Hidden Layer
    keras.layers.Dense(10, activation='sigmoid') # Output Layer
])
-----
